-- ============================================================
-- MIGRATION COMPLETE: Remise à zéro RLS projets/phases/lots
-- Exécuter dans Supabase SQL Editor :
--   https://supabase.com/dashboard > SQL Editor > New Query
--
-- IMPORTANT: Copier-coller TOUT ce fichier et exécuter d'un coup
-- ============================================================

-- =============================================
-- ETAPE 1 : CONTRAINTES UNIQUE (pré-requis)
-- =============================================

-- 1A) DROPPER TOUTES les vieilles contraintes/index UNIQUE sur project_members
--     (sauf la pkey). Notamment project_members_unique_user_idx qui bloque tout.
DO $$
DECLARE
  r record;
BEGIN
  -- Drop toutes les contraintes UNIQUE (contype = 'u')
  FOR r IN
    SELECT conname
    FROM pg_constraint
    WHERE conrelid = 'public.project_members'::regclass
      AND contype = 'u'
  LOOP
    EXECUTE format('ALTER TABLE public.project_members DROP CONSTRAINT IF EXISTS %I', r.conname);
  END LOOP;

  -- Drop aussi les index uniques qui ne sont pas des contraintes
  FOR r IN
    SELECT indexname
    FROM pg_indexes
    WHERE tablename = 'project_members'
      AND schemaname = 'public'
      AND indexdef ILIKE '%unique%'
      AND indexname <> 'project_members_pkey'
  LOOP
    EXECUTE format('DROP INDEX IF EXISTS public.%I', r.indexname);
  END LOOP;
END $$;

-- Drop explicite des noms connus (ceinture + bretelles)
DROP INDEX IF EXISTS public.project_members_unique_user_idx;
DROP INDEX IF EXISTS public.project_members_unique_user_id;

-- 1B) Supprimer doublons project_members
DELETE FROM public.project_members
WHERE id NOT IN (
  SELECT DISTINCT ON (project_id, user_id) id
  FROM public.project_members
  ORDER BY project_id, user_id, accepted_at DESC NULLS LAST
);

-- 1C) Ajouter la bonne contrainte UNIQUE project_members(project_id, user_id)
ALTER TABLE public.project_members
  DROP CONSTRAINT IF EXISTS project_members_unique_project_user;
ALTER TABLE public.project_members
  ADD CONSTRAINT project_members_unique_project_user UNIQUE (project_id, user_id);

-- 1D) DROPPER les vieilles contraintes/index UNIQUE sur phase_members
DO $$
DECLARE
  r record;
BEGIN
  FOR r IN
    SELECT conname
    FROM pg_constraint
    WHERE conrelid = 'public.phase_members'::regclass
      AND contype = 'u'
  LOOP
    EXECUTE format('ALTER TABLE public.phase_members DROP CONSTRAINT IF EXISTS %I', r.conname);
  END LOOP;

  FOR r IN
    SELECT indexname
    FROM pg_indexes
    WHERE tablename = 'phase_members'
      AND schemaname = 'public'
      AND indexdef ILIKE '%unique%'
      AND indexname <> 'phase_members_pkey'
  LOOP
    EXECUTE format('DROP INDEX IF EXISTS public.%I', r.indexname);
  END LOOP;
END $$;

-- 1E) Supprimer doublons phase_members
DELETE FROM public.phase_members
WHERE id NOT IN (
  SELECT DISTINCT ON (phase_id, user_id) id
  FROM public.phase_members
  ORDER BY phase_id, user_id, created_at ASC
);

-- 1F) Ajouter la bonne contrainte UNIQUE phase_members(phase_id, user_id)
ALTER TABLE public.phase_members
  DROP CONSTRAINT IF EXISTS phase_members_phase_id_user_id_key;
ALTER TABLE public.phase_members
  ADD CONSTRAINT phase_members_phase_id_user_id_key UNIQUE (phase_id, user_id);


-- =============================================
-- ETAPE 2 : FONCTIONS HELPERS (CREATE OR REPLACE)
-- =============================================

-- 2A) current_email
CREATE OR REPLACE FUNCTION public.current_email()
RETURNS text
LANGUAGE sql
STABLE
AS $$
  SELECT lower((auth.jwt() ->> 'email')::text);
$$;

-- 2B) is_project_member (membre du projet, tout statut sauf declined/removed)
CREATE OR REPLACE FUNCTION public.is_project_member(p_project_id uuid)
RETURNS boolean
LANGUAGE sql
SECURITY DEFINER
SET search_path = public
AS $$
  SELECT EXISTS (
    SELECT 1
    FROM public.project_members pm
    WHERE pm.project_id = p_project_id
      AND pm.user_id = auth.uid()
      AND lower(btrim(coalesce(pm.status, ''))) NOT IN ('declined', 'removed')
  );
$$;

-- 2C) is_project_invited (invité par email)
CREATE OR REPLACE FUNCTION public.is_project_invited(p_project_id uuid)
RETURNS boolean
LANGUAGE sql
SECURITY DEFINER
SET search_path = public
AS $$
  SELECT EXISTS (
    SELECT 1
    FROM public.project_members pm
    WHERE pm.project_id = p_project_id
      AND lower(coalesce(pm.invited_email, '')) = public.current_email()
  );
$$;

-- 2D) is_project_manager (owner/collaborator + created_by fallback)
CREATE OR REPLACE FUNCTION public.is_project_manager(p_project_id uuid)
RETURNS boolean
LANGUAGE sql
SECURITY DEFINER
SET search_path = public
AS $$
  SELECT EXISTS (
    SELECT 1
    FROM public.project_members pm
    WHERE pm.project_id = p_project_id
      AND pm.user_id = auth.uid()
      AND lower(btrim(coalesce(pm.role, ''))) IN ('owner', 'collaborator', 'pro', 'professionnel')
      AND lower(btrim(coalesce(pm.status, ''))) IN ('accepted', 'active')
  )
  OR EXISTS (
    SELECT 1
    FROM public.projects p
    WHERE p.id = p_project_id
      AND p.created_by = auth.uid()
  )
  OR EXISTS (
    SELECT 1
    FROM public.projects p2
    WHERE p2.id = p_project_id
      AND p2.project_manager_id = auth.uid()
  );
$$;

-- 2E) is_pro
CREATE OR REPLACE FUNCTION public.is_pro()
RETURNS boolean
LANGUAGE sql
SECURITY DEFINER
SET search_path = public
AS $$
  SELECT EXISTS (
    SELECT 1
    FROM public.profiles pr
    WHERE pr.id = auth.uid()
      AND lower(coalesce(pr.user_type, '')) = 'pro'
  );
$$;

-- 2F) is_phase_member
CREATE OR REPLACE FUNCTION public.is_phase_member(p_phase_id uuid)
RETURNS boolean
LANGUAGE sql
SECURITY DEFINER
SET search_path = public
AS $$
  SELECT EXISTS (
    SELECT 1
    FROM public.phase_members pm
    WHERE pm.phase_id = p_phase_id
      AND pm.user_id = auth.uid()
  );
$$;

-- 2G) can_edit_phase
CREATE OR REPLACE FUNCTION public.can_edit_phase(p_phase_id uuid)
RETURNS boolean
LANGUAGE sql
SECURITY DEFINER
SET search_path = public
AS $$
  -- 1. Phase member avec can_edit ou rôle phase_manager
  SELECT EXISTS (
    SELECT 1
    FROM public.phase_members pm
    WHERE pm.phase_id = p_phase_id
      AND pm.user_id = auth.uid()
      AND (pm.can_edit = true OR pm.role = 'phase_manager')
  )
  -- 2. Directement phase_manager sur la phase
  OR EXISTS (
    SELECT 1
    FROM public.phases phm
    WHERE phm.id = p_phase_id
      AND phm.phase_manager_id = auth.uid()
  )
  -- 3. Project manager du projet parent
  OR EXISTS (
    SELECT 1
    FROM public.phases ph
    WHERE ph.id = p_phase_id
      AND public.is_project_manager(ph.project_id)
  );
$$;

-- 2H) can_view_lot
CREATE OR REPLACE FUNCTION public.can_view_lot(p_lot_id uuid)
RETURNS boolean
LANGUAGE sql
SECURITY DEFINER
SET search_path = public
AS $$
  SELECT EXISTS (
    SELECT 1
    FROM public.lots l
    JOIN public.phases ph ON ph.id = l.phase_id
    WHERE l.id = p_lot_id
      AND public.is_project_member(ph.project_id)
  )
  AND (
    -- Project manager voit tout
    EXISTS (
      SELECT 1
      FROM public.lots l2
      JOIN public.phases ph2 ON ph2.id = l2.phase_id
      WHERE l2.id = p_lot_id
        AND public.is_project_manager(ph2.project_id)
    )
    -- Phase manager voit tout dans la phase
    OR EXISTS (
      SELECT 1
      FROM public.lots lpm
      JOIN public.phases phpm ON phpm.id = lpm.phase_id
      WHERE lpm.id = p_lot_id
        AND phpm.phase_manager_id = auth.uid()
        AND public.is_project_member(phpm.project_id)
    )
    -- Phase member voit lots assignés ou tous si autorisé
    OR EXISTS (
      SELECT 1
      FROM public.lots l3
      JOIN public.phase_members pm ON pm.phase_id = l3.phase_id
      WHERE l3.id = p_lot_id
        AND pm.user_id = auth.uid()
        AND (
          pm.can_view_other_lots = true
          OR p_lot_id = ANY(pm.assigned_lots)
          OR pm.role IN ('phase_manager')
        )
    )
    -- Client voit tous les lots en lecture seule
    OR EXISTS (
      SELECT 1
      FROM public.lots l4
      JOIN public.phases ph4 ON ph4.id = l4.phase_id
      JOIN public.project_members pr ON pr.project_id = ph4.project_id
      WHERE l4.id = p_lot_id
        AND pr.user_id = auth.uid()
        AND lower(btrim(coalesce(pr.role,''))) = 'client'
        AND lower(btrim(coalesce(pr.status,''))) IN ('accepted','active')
    )
  );
$$;

-- 2I) can_edit_lot
CREATE OR REPLACE FUNCTION public.can_edit_lot(p_lot_id uuid)
RETURNS boolean
LANGUAGE sql
SECURITY DEFINER
SET search_path = public
AS $$
  -- Project manager peut éditer tous les lots
  SELECT EXISTS (
    SELECT 1
    FROM public.lots l
    JOIN public.phases ph ON ph.id = l.phase_id
    WHERE l.id = p_lot_id
      AND public.is_project_manager(ph.project_id)
  )
  OR EXISTS (
    SELECT 1
    FROM public.lots lpm
    JOIN public.phases phpm ON phpm.id = lpm.phase_id
    WHERE lpm.id = p_lot_id
      AND phpm.phase_manager_id = auth.uid()
      AND public.is_project_member(phpm.project_id)
  )
  OR EXISTS (
    SELECT 1
    FROM public.lots l
    JOIN public.phase_members pm ON pm.phase_id = l.phase_id
    WHERE l.id = p_lot_id
      AND pm.user_id = auth.uid()
      AND (pm.can_edit = true OR pm.role = 'phase_manager')
      AND (
        pm.role <> 'entreprise'
        OR l.id = ANY(pm.assigned_lots)
      )
  );
$$;


-- =============================================
-- ETAPE 3 : TRIGGER ensure_phase_manager_membership
-- =============================================

CREATE OR REPLACE FUNCTION public.ensure_phase_manager_membership()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
  IF NEW.phase_manager_id IS NULL THEN
    RETURN NEW;
  END IF;

  INSERT INTO public.phase_members (phase_id, user_id, role, can_edit, can_view_other_lots)
  VALUES (NEW.id, NEW.phase_manager_id, 'phase_manager', true, true)
  ON CONFLICT (phase_id, user_id) DO UPDATE
    SET role = 'phase_manager', can_edit = true, can_view_other_lots = true;

  RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS phases_ensure_phase_manager_membership ON public.phases;
CREATE TRIGGER phases_ensure_phase_manager_membership
  AFTER INSERT OR UPDATE OF phase_manager_id ON public.phases
  FOR EACH ROW
  EXECUTE FUNCTION public.ensure_phase_manager_membership();


-- =============================================
-- ETAPE 4 : ENABLE RLS sur toutes les tables
-- =============================================

ALTER TABLE public.projects ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.project_members ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.project_messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.phases ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.phase_members ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.lots ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.lot_tasks ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.quotes ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.invoices ENABLE ROW LEVEL SECURITY;


-- =============================================
-- ETAPE 5 : POLICIES (DROP + CREATE pour chaque)
-- =============================================

-- ===== PROJECTS =====
DROP POLICY IF EXISTS projects_select ON public.projects;
CREATE POLICY projects_select ON public.projects
  FOR SELECT USING (
    public.is_project_member(id)
    OR public.is_project_invited(id)
    OR created_by = auth.uid()
  );

DROP POLICY IF EXISTS projects_insert ON public.projects;
CREATE POLICY projects_insert ON public.projects
  FOR INSERT WITH CHECK (
    public.is_pro()
    AND created_by = auth.uid()
  );

DROP POLICY IF EXISTS projects_update ON public.projects;
CREATE POLICY projects_update ON public.projects
  FOR UPDATE
  USING (public.is_project_manager(id))
  WITH CHECK (public.is_project_manager(id));

DROP POLICY IF EXISTS projects_delete ON public.projects;
CREATE POLICY projects_delete ON public.projects
  FOR DELETE USING (public.is_project_manager(id));

-- ===== PROJECT_MEMBERS =====
DROP POLICY IF EXISTS project_members_select ON public.project_members;
CREATE POLICY project_members_select ON public.project_members
  FOR SELECT USING (
    public.is_project_member(project_id)
    OR public.is_project_invited(project_id)
  );

DROP POLICY IF EXISTS project_members_insert ON public.project_members;
CREATE POLICY project_members_insert ON public.project_members
  FOR INSERT WITH CHECK (
    public.is_project_manager(project_id)
  );

DROP POLICY IF EXISTS project_members_update ON public.project_members;
CREATE POLICY project_members_update ON public.project_members
  FOR UPDATE
  USING (
    public.is_project_manager(project_id)
    OR user_id = auth.uid()
    OR lower(coalesce(invited_email, '')) = public.current_email()
  )
  WITH CHECK (
    public.is_project_manager(project_id)
    OR user_id = auth.uid()
    OR lower(coalesce(invited_email, '')) = public.current_email()
  );

DROP POLICY IF EXISTS project_members_delete ON public.project_members;
CREATE POLICY project_members_delete ON public.project_members
  FOR DELETE USING (public.is_project_manager(project_id));

-- ===== PROJECT_MESSAGES =====
DROP POLICY IF EXISTS project_messages_select ON public.project_messages;
CREATE POLICY project_messages_select ON public.project_messages
  FOR SELECT USING (public.is_project_member(project_id));

DROP POLICY IF EXISTS project_messages_insert ON public.project_messages;
CREATE POLICY project_messages_insert ON public.project_messages
  FOR INSERT WITH CHECK (
    public.is_project_member(project_id)
    AND sender_id = auth.uid()
  );

DROP POLICY IF EXISTS project_messages_delete ON public.project_messages;
CREATE POLICY project_messages_delete ON public.project_messages
  FOR DELETE USING (public.is_project_manager(project_id));

-- ===== PHASES =====
DROP POLICY IF EXISTS phases_select ON public.phases;
CREATE POLICY phases_select ON public.phases
  FOR SELECT USING (
    public.is_project_member(project_id)
    OR public.is_project_invited(project_id)
    OR public.is_project_manager(project_id)
  );

DROP POLICY IF EXISTS phases_insert ON public.phases;
CREATE POLICY phases_insert ON public.phases
  FOR INSERT WITH CHECK (public.is_project_manager(project_id));

DROP POLICY IF EXISTS phases_update ON public.phases;
CREATE POLICY phases_update ON public.phases
  FOR UPDATE
  USING (public.can_edit_phase(id))
  WITH CHECK (public.can_edit_phase(id));

DROP POLICY IF EXISTS phases_delete ON public.phases;
CREATE POLICY phases_delete ON public.phases
  FOR DELETE USING (public.can_edit_phase(id));

-- ===== PHASE_MEMBERS =====
DROP POLICY IF EXISTS phase_members_select ON public.phase_members;
CREATE POLICY phase_members_select ON public.phase_members
  FOR SELECT USING (
    user_id = auth.uid()
    OR EXISTS (
      SELECT 1 FROM public.phases ph
      WHERE ph.id = phase_id AND public.is_project_manager(ph.project_id)
    )
  );

DROP POLICY IF EXISTS phase_members_insert ON public.phase_members;
CREATE POLICY phase_members_insert ON public.phase_members
  FOR INSERT WITH CHECK (
    EXISTS (
      SELECT 1 FROM public.phases ph
      WHERE ph.id = phase_id AND public.is_project_manager(ph.project_id)
    )
  );

DROP POLICY IF EXISTS phase_members_update ON public.phase_members;
CREATE POLICY phase_members_update ON public.phase_members
  FOR UPDATE
  USING (
    EXISTS (
      SELECT 1 FROM public.phases ph
      WHERE ph.id = phase_id AND public.is_project_manager(ph.project_id)
    )
  )
  WITH CHECK (
    EXISTS (
      SELECT 1 FROM public.phases ph
      WHERE ph.id = phase_id AND public.is_project_manager(ph.project_id)
    )
  );

DROP POLICY IF EXISTS phase_members_delete ON public.phase_members;
CREATE POLICY phase_members_delete ON public.phase_members
  FOR DELETE USING (
    EXISTS (
      SELECT 1 FROM public.phases ph
      WHERE ph.id = phase_id AND public.is_project_manager(ph.project_id)
    )
  );

-- ===== LOTS =====
DROP POLICY IF EXISTS lots_select ON public.lots;
CREATE POLICY lots_select ON public.lots
  FOR SELECT USING (public.can_view_lot(id));

DROP POLICY IF EXISTS lots_insert ON public.lots;
CREATE POLICY lots_insert ON public.lots
  FOR INSERT WITH CHECK (public.can_edit_phase(phase_id));

DROP POLICY IF EXISTS lots_update ON public.lots;
CREATE POLICY lots_update ON public.lots
  FOR UPDATE
  USING (public.can_edit_lot(id))
  WITH CHECK (public.can_edit_lot(id));

DROP POLICY IF EXISTS lots_delete ON public.lots;
CREATE POLICY lots_delete ON public.lots
  FOR DELETE USING (public.can_edit_lot(id));

-- ===== LOT_TASKS =====
DROP POLICY IF EXISTS lot_tasks_select ON public.lot_tasks;
CREATE POLICY lot_tasks_select ON public.lot_tasks
  FOR SELECT USING (public.can_view_lot(lot_id));

DROP POLICY IF EXISTS lot_tasks_insert ON public.lot_tasks;
CREATE POLICY lot_tasks_insert ON public.lot_tasks
  FOR INSERT WITH CHECK (public.can_edit_lot(lot_id));

DROP POLICY IF EXISTS lot_tasks_update ON public.lot_tasks;
CREATE POLICY lot_tasks_update ON public.lot_tasks
  FOR UPDATE
  USING (public.can_edit_lot(lot_id))
  WITH CHECK (public.can_edit_lot(lot_id));

DROP POLICY IF EXISTS lot_tasks_delete ON public.lot_tasks;
CREATE POLICY lot_tasks_delete ON public.lot_tasks
  FOR DELETE USING (public.can_edit_lot(lot_id));

-- ===== QUOTES =====
DROP POLICY IF EXISTS quotes_select ON public.quotes;
CREATE POLICY quotes_select ON public.quotes
  FOR SELECT USING (public.can_view_lot(lot_id));

DROP POLICY IF EXISTS quotes_write ON public.quotes;
CREATE POLICY quotes_write ON public.quotes
  FOR ALL
  USING (public.can_edit_lot(lot_id))
  WITH CHECK (public.can_edit_lot(lot_id));

-- ===== INVOICES =====
DROP POLICY IF EXISTS invoices_select ON public.invoices;
CREATE POLICY invoices_select ON public.invoices
  FOR SELECT USING (public.can_view_lot(lot_id));

DROP POLICY IF EXISTS invoices_write ON public.invoices;
CREATE POLICY invoices_write ON public.invoices
  FOR ALL
  USING (public.can_edit_lot(lot_id))
  WITH CHECK (public.can_edit_lot(lot_id));


-- =============================================
-- ETAPE 6 : CORRECTION DES DONNÉES EXISTANTES
-- =============================================

-- 6A) Ajouter le créateur de chaque projet dans project_members s'il manque
INSERT INTO public.project_members (project_id, user_id, role, status, invited_by, accepted_at)
SELECT
  p.id AS project_id,
  p.created_by AS user_id,
  'owner' AS role,
  'accepted' AS status,
  p.created_by AS invited_by,
  now() AS accepted_at
FROM public.projects p
WHERE p.created_by IS NOT NULL
  AND NOT EXISTS (
    SELECT 1 FROM public.project_members pm
    WHERE pm.project_id = p.id AND pm.user_id = p.created_by
  )
ON CONFLICT (project_id, user_id) DO NOTHING;

-- 6B) Mettre à jour les phases sans phase_manager_id (assigner le owner du projet)
UPDATE public.phases ph
SET phase_manager_id = (
  SELECT pm.user_id
  FROM public.project_members pm
  WHERE pm.project_id = ph.project_id
    AND lower(btrim(coalesce(pm.role, ''))) = 'owner'
    AND lower(btrim(coalesce(pm.status, ''))) IN ('accepted', 'active')
  LIMIT 1
)
WHERE ph.phase_manager_id IS NULL;

-- Si toujours NULL, assigner le created_by du projet
UPDATE public.phases ph
SET phase_manager_id = (
  SELECT p.created_by FROM public.projects p WHERE p.id = ph.project_id
)
WHERE ph.phase_manager_id IS NULL;

-- 6C) Ajouter phase_members pour les owners/collaborators de projets
INSERT INTO public.phase_members (phase_id, user_id, role, can_edit, can_view_other_lots)
SELECT
  ph.id AS phase_id,
  pm.user_id,
  'phase_manager' AS role,
  true AS can_edit,
  true AS can_view_other_lots
FROM public.phases ph
JOIN public.project_members pm
  ON pm.project_id = ph.project_id
  AND lower(btrim(coalesce(pm.role, ''))) IN ('owner', 'collaborator')
  AND lower(btrim(coalesce(pm.status, ''))) IN ('accepted', 'active')
WHERE NOT EXISTS (
  SELECT 1 FROM public.phase_members existing
  WHERE existing.phase_id = ph.id AND existing.user_id = pm.user_id
)
ON CONFLICT (phase_id, user_id) DO NOTHING;

-- 6D) Ajouter le created_by du projet dans phase_members si absent
INSERT INTO public.phase_members (phase_id, user_id, role, can_edit, can_view_other_lots)
SELECT
  ph.id AS phase_id,
  p.created_by AS user_id,
  'phase_manager' AS role,
  true AS can_edit,
  true AS can_view_other_lots
FROM public.phases ph
JOIN public.projects p ON p.id = ph.project_id
WHERE p.created_by IS NOT NULL
  AND NOT EXISTS (
    SELECT 1 FROM public.phase_members existing
    WHERE existing.phase_id = ph.id AND existing.user_id = p.created_by
  )
ON CONFLICT (phase_id, user_id) DO NOTHING;

-- 6E) Ajouter le phase_manager_id dans phase_members si absent
INSERT INTO public.phase_members (phase_id, user_id, role, can_edit, can_view_other_lots)
SELECT
  ph.id AS phase_id,
  ph.phase_manager_id AS user_id,
  'phase_manager' AS role,
  true AS can_edit,
  true AS can_view_other_lots
FROM public.phases ph
WHERE ph.phase_manager_id IS NOT NULL
  AND NOT EXISTS (
    SELECT 1 FROM public.phase_members existing
    WHERE existing.phase_id = ph.id AND existing.user_id = ph.phase_manager_id
  )
ON CONFLICT (phase_id, user_id) DO NOTHING;


-- =============================================
-- ETAPE 7 : FONCTIONS RPC SECURITY DEFINER
-- Bypass total du RLS pour les opérations d'écriture.
-- Les permissions sont vérifiées dans chaque fonction.
-- =============================================

-- --------- Helper interne : l'utilisateur peut-il écrire dans cette phase ? ---------
CREATE OR REPLACE FUNCTION public._can_write_in_phase(p_phase_id uuid, p_user_id uuid)
RETURNS boolean
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
  v_project_id uuid;
BEGIN
  -- Récupérer le project_id de la phase
  SELECT project_id INTO v_project_id
  FROM public.phases WHERE id = p_phase_id;
  IF v_project_id IS NULL THEN
    RETURN false;
  END IF;

  -- 1. Chef de projet : créateur du projet
  IF EXISTS (SELECT 1 FROM public.projects WHERE id = v_project_id AND created_by = p_user_id) THEN
    RETURN true;
  END IF;

  -- 2. Chef de projet : project_manager_id
  IF EXISTS (SELECT 1 FROM public.projects WHERE id = v_project_id AND project_manager_id = p_user_id) THEN
    RETURN true;
  END IF;

  -- 3. Chef de projet : owner/collaborator dans project_members
  IF EXISTS (
    SELECT 1 FROM public.project_members
    WHERE project_id = v_project_id
      AND user_id = p_user_id
      AND role IN ('owner', 'collaborator', 'pro', 'professionnel', 'chef_de_projet')
      AND status IN ('accepted', 'active')
  ) THEN
    RETURN true;
  END IF;

  -- 4. Responsable de phase : phase_manager_id
  IF EXISTS (SELECT 1 FROM public.phases WHERE id = p_phase_id AND phase_manager_id = p_user_id) THEN
    RETURN true;
  END IF;

  -- 5. Responsable de phase : phase_member avec can_edit ou rôle phase_manager
  IF EXISTS (
    SELECT 1 FROM public.phase_members
    WHERE phase_id = p_phase_id
      AND user_id = p_user_id
      AND (can_edit = true OR role IN ('phase_manager', 'responsable_phase'))
  ) THEN
    RETURN true;
  END IF;

  -- Sinon : pas autorisé (client, observateur, etc.)
  RETURN false;
END;
$$;

-- --------- RPC : Créer un lot ---------
CREATE OR REPLACE FUNCTION public.rpc_create_lot(
  p_phase_id uuid,
  p_name text,
  p_description text DEFAULT NULL,
  p_lot_type text DEFAULT NULL,
  p_company_name text DEFAULT NULL,
  p_company_contact_name text DEFAULT NULL,
  p_company_contact_email text DEFAULT NULL,
  p_company_contact_phone text DEFAULT NULL,
  p_responsible_user_id uuid DEFAULT NULL,
  p_start_date text DEFAULT NULL,
  p_end_date text DEFAULT NULL,
  p_budget_estimated numeric DEFAULT 0,
  p_status text DEFAULT 'planifie'
)
RETURNS uuid
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
  v_user_id uuid;
  v_lot_id uuid;
BEGIN
  v_user_id := auth.uid();
  IF v_user_id IS NULL THEN
    RAISE EXCEPTION 'Non authentifie';
  END IF;

  IF NOT public._can_write_in_phase(p_phase_id, v_user_id) THEN
    RAISE EXCEPTION 'Acces refuse : vous n''etes pas chef de projet ni responsable de cette phase.';
  END IF;

  INSERT INTO public.lots (
    phase_id, name, description, lot_type,
    company_name, company_contact_name, company_contact_email, company_contact_phone,
    responsible_user_id, start_date, end_date,
    budget_estimated, status
  ) VALUES (
    p_phase_id,
    btrim(p_name),
    btrim(p_description),
    btrim(p_lot_type),
    btrim(p_company_name),
    btrim(p_company_contact_name),
    btrim(p_company_contact_email),
    btrim(p_company_contact_phone),
    p_responsible_user_id,
    p_start_date::date,
    p_end_date::date,
    p_budget_estimated,
    coalesce(p_status, 'planifie')
  )
  RETURNING id INTO v_lot_id;

  RETURN v_lot_id;
END;
$$;

-- --------- RPC : Modifier un lot ---------
CREATE OR REPLACE FUNCTION public.rpc_update_lot(
  p_lot_id uuid,
  p_name text DEFAULT NULL,
  p_description text DEFAULT NULL,
  p_lot_type text DEFAULT NULL,
  p_company_name text DEFAULT NULL,
  p_company_contact_name text DEFAULT NULL,
  p_company_contact_email text DEFAULT NULL,
  p_company_contact_phone text DEFAULT NULL,
  p_responsible_user_id uuid DEFAULT NULL,
  p_start_date text DEFAULT NULL,
  p_end_date text DEFAULT NULL,
  p_budget_estimated numeric DEFAULT NULL,
  p_budget_actual numeric DEFAULT NULL,
  p_status text DEFAULT NULL,
  p_progress_percentage numeric DEFAULT NULL
)
RETURNS void
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
  v_user_id uuid;
  v_phase_id uuid;
BEGIN
  v_user_id := auth.uid();
  IF v_user_id IS NULL THEN
    RAISE EXCEPTION 'Non authentifie';
  END IF;

  SELECT phase_id INTO v_phase_id FROM public.lots WHERE id = p_lot_id;
  IF v_phase_id IS NULL THEN
    RAISE EXCEPTION 'Lot introuvable';
  END IF;

  IF NOT public._can_write_in_phase(v_phase_id, v_user_id) THEN
    RAISE EXCEPTION 'Acces refuse : vous n''etes pas chef de projet ni responsable de cette phase.';
  END IF;

  UPDATE public.lots SET
    name                  = coalesce(btrim(p_name), name),
    description           = CASE WHEN p_description IS NOT NULL THEN btrim(p_description) ELSE description END,
    lot_type              = CASE WHEN p_lot_type IS NOT NULL THEN btrim(p_lot_type) ELSE lot_type END,
    company_name          = CASE WHEN p_company_name IS NOT NULL THEN btrim(p_company_name) ELSE company_name END,
    company_contact_name  = CASE WHEN p_company_contact_name IS NOT NULL THEN btrim(p_company_contact_name) ELSE company_contact_name END,
    company_contact_email = CASE WHEN p_company_contact_email IS NOT NULL THEN btrim(p_company_contact_email) ELSE company_contact_email END,
    company_contact_phone = CASE WHEN p_company_contact_phone IS NOT NULL THEN btrim(p_company_contact_phone) ELSE company_contact_phone END,
    responsible_user_id   = CASE WHEN p_responsible_user_id IS NOT NULL THEN p_responsible_user_id ELSE responsible_user_id END,
    start_date            = CASE WHEN p_start_date IS NOT NULL THEN p_start_date::date ELSE start_date END,
    end_date              = CASE WHEN p_end_date IS NOT NULL THEN p_end_date::date ELSE end_date END,
    budget_estimated      = CASE WHEN p_budget_estimated IS NOT NULL THEN p_budget_estimated ELSE budget_estimated END,
    budget_actual         = CASE WHEN p_budget_actual IS NOT NULL THEN p_budget_actual ELSE budget_actual END,
    status                = CASE WHEN p_status IS NOT NULL THEN p_status ELSE status END,
    progress_percentage   = CASE WHEN p_progress_percentage IS NOT NULL THEN p_progress_percentage ELSE progress_percentage END,
    updated_at            = now()
  WHERE id = p_lot_id;
END;
$$;

-- --------- RPC : Supprimer un lot ---------
CREATE OR REPLACE FUNCTION public.rpc_delete_lot(p_lot_id uuid)
RETURNS void
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
  v_user_id uuid;
  v_phase_id uuid;
BEGIN
  v_user_id := auth.uid();
  IF v_user_id IS NULL THEN
    RAISE EXCEPTION 'Non authentifie';
  END IF;

  SELECT phase_id INTO v_phase_id FROM public.lots WHERE id = p_lot_id;
  IF v_phase_id IS NULL THEN
    RAISE EXCEPTION 'Lot introuvable';
  END IF;

  IF NOT public._can_write_in_phase(v_phase_id, v_user_id) THEN
    RAISE EXCEPTION 'Acces refuse : vous n''etes pas chef de projet ni responsable de cette phase.';
  END IF;

  -- Supprimer les dépendances d'abord
  DELETE FROM public.lot_tasks WHERE lot_id = p_lot_id;
  DELETE FROM public.quotes WHERE lot_id = p_lot_id;
  DELETE FROM public.invoices WHERE lot_id = p_lot_id;
  DELETE FROM public.lots WHERE id = p_lot_id;
END;
$$;

-- --------- GRANT : permettre aux utilisateurs authentifiés d'appeler les RPCs ---------
GRANT EXECUTE ON FUNCTION public._can_write_in_phase(uuid, uuid) TO authenticated;
GRANT EXECUTE ON FUNCTION public.rpc_create_lot(uuid, text, text, text, text, text, text, text, uuid, text, text, numeric, text) TO authenticated;
GRANT EXECUTE ON FUNCTION public.rpc_update_lot(uuid, text, text, text, text, text, text, text, uuid, text, text, numeric, numeric, text, numeric) TO authenticated;
GRANT EXECUTE ON FUNCTION public.rpc_delete_lot(uuid) TO authenticated;


-- =============================================
-- ETAPE 8 : RELOAD SCHEMA CACHE
-- =============================================
NOTIFY pgrst, 'reload schema';
