"""Feedback API endpoints for user rating system."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

from agent.logging_config import logger
from agent.monitoring import track_request
from agent.supabase_client import get_client

router = APIRouter()


class FeedbackInput(BaseModel):
    """Input model for submitting feedback."""

    conversation_id: str = Field(..., min_length=1, description="ID de la conversation")
    message_id: Optional[str] = Field(None, description="ID du message spécifique (optionnel)")
    rating: int = Field(..., ge=1, le=5, description="Note de 1 à 5 (1=mauvais, 5=excellent)")
    rating_type: Literal["stars", "thumbs"] = Field(
        default="stars", description="Type de rating: stars (1-5) ou thumbs (1=down, 5=up)"
    )
    comment: Optional[str] = Field(None, max_length=1000, description="Commentaire optionnel")
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Métadonnées du message (intent, route, etc.)"
    )
    user_id: Optional[str] = Field(None, description="ID de l'utilisateur (optionnel)")
    user_role: Optional[str] = Field(None, description="Rôle de l'utilisateur (particulier/professionnel)")

    @validator("rating")
    def validate_rating_for_type(cls, v, values):
        """Validate rating based on rating_type."""
        rating_type = values.get("rating_type", "stars")
        if rating_type == "thumbs" and v not in [1, 5]:
            raise ValueError("For thumbs rating, use 1 (thumbs down) or 5 (thumbs up)")
        return v


class FeedbackResponse(BaseModel):
    """Response model after submitting feedback."""

    id: str
    conversation_id: str
    message_id: Optional[str]
    rating: int
    rating_type: str
    comment: Optional[str]
    metadata: Dict[str, Any]
    user_id: Optional[str]
    user_role: Optional[str]
    created_at: datetime
    updated_at: datetime


class FeedbackAnalytics(BaseModel):
    """Analytics summary for feedbacks."""

    total_feedbacks: int = Field(..., description="Nombre total de feedbacks")
    average_rating: float = Field(..., description="Note moyenne (1-5)")
    positive_count: int = Field(..., description="Nombre de feedbacks positifs (≥4)")
    negative_count: int = Field(..., description="Nombre de feedbacks négatifs (≤2)")
    neutral_count: int = Field(..., description="Nombre de feedbacks neutres (3)")
    with_comment_count: int = Field(..., description="Nombre de feedbacks avec commentaire")

    # Distribution par rating
    rating_distribution: Dict[str, int] = Field(
        default_factory=dict, description="Distribution des ratings {1: count, 2: count, ...}"
    )

    # Analytics par intent
    by_intent: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, description="Stats par intent {intent: {avg_rating, count}}"
    )

    # Analytics par route
    by_route: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, description="Stats par route {route: {avg_rating, count}}"
    )

    # Feedbacks récents mal notés
    low_rated_feedbacks: List[Dict[str, Any]] = Field(
        default_factory=list, description="Liste des feedbacks mal notés (≤2)"
    )


@router.post("/feedback", response_model=FeedbackResponse)
@track_request("feedback_submit")
async def submit_feedback(payload: FeedbackInput, request: Request):
    """Submit feedback for an assistant response.

    Allows users to rate responses with stars (1-5) or thumbs (up/down),
    and optionally provide a comment.

    Rate limit: Inherited from global limit (100/minute).
    """
    sb = get_client()
    if not sb:
        raise HTTPException(status_code=503, detail="Database connection unavailable")

    try:
        # Prepare feedback data
        feedback_data = {
            "conversation_id": payload.conversation_id,
            "message_id": payload.message_id,
            "rating": payload.rating,
            "rating_type": payload.rating_type,
            "comment": payload.comment,
            "metadata": payload.metadata or {},
            "user_id": payload.user_id,
            "user_role": payload.user_role,
        }

        # Insert feedback into Supabase
        response = sb.table("feedbacks").insert(feedback_data).execute()

        if not response.data:
            raise HTTPException(status_code=500, detail="Failed to insert feedback")

        feedback = response.data[0]

        logger.info(
            "Feedback submitted: conversation_id=%s, rating=%d/%s, has_comment=%s",
            payload.conversation_id,
            payload.rating,
            payload.rating_type,
            bool(payload.comment),
        )

        return FeedbackResponse(**feedback)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error submitting feedback: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to submit feedback: {str(exc)}")


@router.get("/feedback/{conversation_id}", response_model=List[FeedbackResponse])
async def get_feedback_by_conversation(conversation_id: str):
    """Get all feedbacks for a specific conversation.

    Args:
        conversation_id: The conversation ID to fetch feedbacks for
    """
    sb = get_client()
    if not sb:
        raise HTTPException(status_code=503, detail="Database connection unavailable")

    try:
        response = (
            sb.table("feedbacks")
            .select("*")
            .eq("conversation_id", conversation_id)
            .order("created_at", desc=True)
            .execute()
        )

        feedbacks = [FeedbackResponse(**fb) for fb in response.data]
        return feedbacks

    except Exception as exc:
        logger.error("Error fetching feedbacks: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch feedbacks: {str(exc)}")


@router.get("/analytics/feedback", response_model=FeedbackAnalytics)
@track_request("feedback_analytics")
async def get_feedback_analytics(
    days: int = 30,
    intent: Optional[str] = None,
    route: Optional[str] = None,
):
    """Get feedback analytics and statistics.

    Args:
        days: Number of days to include in analytics (default: 30)
        intent: Filter by specific intent (optional)
        route: Filter by specific route (optional)

    Returns:
        Comprehensive analytics including average rating, distribution, and insights
    """
    sb = get_client()
    if not sb:
        raise HTTPException(status_code=503, detail="Database connection unavailable")

    try:
        # Build query with filters
        query = sb.table("feedbacks").select("*")

        # Filter by date range
        if days:
            from datetime import timedelta
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
            query = query.gte("created_at", cutoff_date)

        # Filter by intent
        if intent:
            query = query.eq("metadata->>intent", intent)

        # Filter by route
        if route:
            query = query.eq("metadata->>route", route)

        response = query.execute()
        feedbacks = response.data

        if not feedbacks:
            return FeedbackAnalytics(
                total_feedbacks=0,
                average_rating=0.0,
                positive_count=0,
                negative_count=0,
                neutral_count=0,
                with_comment_count=0,
                rating_distribution={},
                by_intent={},
                by_route={},
                low_rated_feedbacks=[],
            )

        # Calculate statistics
        total = len(feedbacks)
        ratings = [fb["rating"] for fb in feedbacks]
        avg_rating = sum(ratings) / total if total > 0 else 0.0

        positive = sum(1 for r in ratings if r >= 4)
        negative = sum(1 for r in ratings if r <= 2)
        neutral = sum(1 for r in ratings if r == 3)
        with_comment = sum(1 for fb in feedbacks if fb.get("comment"))

        # Rating distribution
        rating_dist = {}
        for r in range(1, 6):
            rating_dist[str(r)] = sum(1 for rating in ratings if rating == r)

        # Analytics by intent
        by_intent: Dict[str, Dict[str, Any]] = {}
        for fb in feedbacks:
            fb_intent = fb.get("metadata", {}).get("intent")
            if fb_intent:
                if fb_intent not in by_intent:
                    by_intent[fb_intent] = {"ratings": [], "count": 0}
                by_intent[fb_intent]["ratings"].append(fb["rating"])
                by_intent[fb_intent]["count"] += 1

        # Calculate average for each intent
        for intent_name, data in by_intent.items():
            data["avg_rating"] = round(sum(data["ratings"]) / data["count"], 2)
            del data["ratings"]  # Remove raw ratings

        # Analytics by route
        by_route: Dict[str, Dict[str, Any]] = {}
        for fb in feedbacks:
            fb_route = fb.get("metadata", {}).get("route")
            if fb_route:
                if fb_route not in by_route:
                    by_route[fb_route] = {"ratings": [], "count": 0}
                by_route[fb_route]["ratings"].append(fb["rating"])
                by_route[fb_route]["count"] += 1

        # Calculate average for each route
        for route_name, data in by_route.items():
            data["avg_rating"] = round(sum(data["ratings"]) / data["count"], 2)
            del data["ratings"]  # Remove raw ratings

        # Low rated feedbacks (≤2)
        low_rated = [
            {
                "id": fb["id"],
                "conversation_id": fb["conversation_id"],
                "rating": fb["rating"],
                "comment": fb.get("comment"),
                "metadata": fb.get("metadata", {}),
                "created_at": fb["created_at"],
            }
            for fb in feedbacks
            if fb["rating"] <= 2
        ]
        low_rated = sorted(low_rated, key=lambda x: x["created_at"], reverse=True)[:10]

        analytics = FeedbackAnalytics(
            total_feedbacks=total,
            average_rating=round(avg_rating, 2),
            positive_count=positive,
            negative_count=negative,
            neutral_count=neutral,
            with_comment_count=with_comment,
            rating_distribution=rating_dist,
            by_intent=by_intent,
            by_route=by_route,
            low_rated_feedbacks=low_rated,
        )

        logger.info(
            "Feedback analytics: total=%d, avg_rating=%.2f, positive=%d, negative=%d",
            total,
            avg_rating,
            positive,
            negative,
        )

        return analytics

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error fetching feedback analytics: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch analytics: {str(exc)}")


@router.delete("/feedback/{feedback_id}")
async def delete_feedback(feedback_id: str):
    """Delete a specific feedback by ID.

    Args:
        feedback_id: The UUID of the feedback to delete

    Returns:
        Success message
    """
    sb = get_client()
    if not sb:
        raise HTTPException(status_code=503, detail="Database connection unavailable")

    try:
        response = sb.table("feedbacks").delete().eq("id", feedback_id).execute()

        if not response.data:
            raise HTTPException(status_code=404, detail="Feedback not found")

        logger.info("Feedback deleted: id=%s", feedback_id)

        return JSONResponse({"message": "Feedback deleted successfully", "id": feedback_id})

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error deleting feedback: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete feedback: {str(exc)}")
