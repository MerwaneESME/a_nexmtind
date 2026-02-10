-- Exécuter dans Supabase SQL Editor : https://supabase.com/dashboard > SQL Editor > New Query

-- ============================================
-- Table: task_learning_stats
-- Utilisée par l'agent IA pour estimer les durées de tâches BTP dans les plannings.
-- Requêtée par le endpoint /project-chat via _build_scoped_project_context.
-- ============================================

CREATE TABLE IF NOT EXISTS public.task_learning_stats (
  id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
  normalized_label text NOT NULL,
  example_name text,
  avg_duration_hours numeric,
  avg_start_hour numeric,
  avg_end_hour numeric,
  sample_count integer DEFAULT 0,
  trade text NOT NULL,
  created_at timestamptz DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_tls_trade ON public.task_learning_stats(trade);
CREATE INDEX IF NOT EXISTS idx_tls_label ON public.task_learning_stats(normalized_label);

-- Seed avec données réalistes BTP (durées moyennes constatées)
INSERT INTO public.task_learning_stats
  (normalized_label, example_name, avg_duration_hours, avg_start_hour, avg_end_hour, sample_count, trade)
VALUES
  ('demolition', 'Démolition cloisons', 16, 8, 17, 45, 'Gros oeuvre'),
  ('terrassement', 'Terrassement fondations', 24, 7, 17, 30, 'Gros oeuvre'),
  ('maconnerie', 'Montage parpaings', 40, 7, 17, 55, 'Gros oeuvre'),
  ('fondations', 'Coulage fondations', 16, 7, 16, 28, 'Gros oeuvre'),
  ('charpente', 'Pose charpente traditionnelle', 32, 7, 17, 25, 'Charpente'),
  ('couverture', 'Couverture tuiles', 24, 8, 17, 35, 'Couverture'),
  ('placo', 'Pose placo BA13', 24, 8, 17, 120, 'Platrerie'),
  ('enduit', 'Enduit plâtre intérieur', 16, 8, 17, 60, 'Platrerie'),
  ('isolation', 'Isolation laine de verre', 16, 8, 17, 90, 'Isolation'),
  ('peinture', 'Peinture 2 couches', 16, 8, 16, 89, 'Peinture'),
  ('electricite', 'Rénovation tableau', 8, 8, 17, 67, 'Electricite'),
  ('plomberie', 'Remplacement sanitaires', 12, 8, 16, 55, 'Plomberie'),
  ('carrelage', 'Pose carrelage sol', 20, 8, 17, 78, 'Carrelage'),
  ('menuiserie', 'Pose fenêtres PVC', 8, 8, 17, 42, 'Menuiserie'),
  ('extension', 'Extension ossature bois', 160, 7, 18, 12, 'Extension'),
  ('ravalement', 'Ravalement façade', 40, 8, 17, 20, 'Ravalement'),
  ('etancheite', 'Étanchéité toiture terrasse', 16, 8, 17, 15, 'Etancheite'),
  ('climatisation', 'Installation climatisation', 8, 8, 17, 22, 'CVC'),
  ('chauffage', 'Pose radiateurs', 12, 8, 17, 38, 'CVC')
ON CONFLICT DO NOTHING;

-- RLS (lecture publique)
ALTER TABLE public.task_learning_stats ENABLE ROW LEVEL SECURITY;
CREATE POLICY IF NOT EXISTS "Allow read for all" ON public.task_learning_stats FOR SELECT USING (true);
