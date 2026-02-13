-- Exécuter dans Supabase SQL Editor : https://supabase.com/dashboard > SQL Editor > New Query
-- Schéma complet disponible dans : output/supabase_feedbacks_schema.sql

-- ============================================
-- Table: feedbacks
-- Utilisée par l'agent IA pour collecter les retours utilisateurs.
-- Requêtée par api/feedback.py.
-- ============================================

CREATE TABLE IF NOT EXISTS public.feedbacks (
  id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
  conversation_id text NOT NULL,
  message_id text,
  rating integer NOT NULL CHECK (rating >= 1 AND rating <= 5),
  rating_type text NOT NULL DEFAULT 'thumbs' CHECK (rating_type IN ('thumbs', 'stars')),
  comment text,
  metadata jsonb DEFAULT '{}',
  user_id text,
  user_role text,
  created_at timestamptz DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_fb_conv ON public.feedbacks(conversation_id);
CREATE INDEX IF NOT EXISTS idx_fb_created ON public.feedbacks(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_fb_rating ON public.feedbacks(rating);

ALTER TABLE public.feedbacks ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Allow insert for all" ON public.feedbacks;
CREATE POLICY "Allow insert for all" ON public.feedbacks FOR INSERT WITH CHECK (true);
DROP POLICY IF EXISTS "Allow read for all" ON public.feedbacks;
CREATE POLICY "Allow read for all" ON public.feedbacks FOR SELECT USING (true);
DROP POLICY IF EXISTS "Allow delete for all" ON public.feedbacks;
CREATE POLICY "Allow delete for all" ON public.feedbacks FOR DELETE USING (true);
