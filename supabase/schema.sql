-- Supabase schema for BTP devis/factures and related entities
-- Context only; apply via Supabase migration tools as needed.

create extension if not exists "uuid-ossp";

create table public.clients (
  id uuid primary key default gen_random_uuid(),
  name text not null,
  contact jsonb,
  address text,
  created_at timestamptz default now()
);

create table public.profiles (
  id uuid primary key,
  email text,
  full_name text,
  phone text,
  company_name text,
  siret text,
  user_type text,
  avatar_url text,
  address text,
  city text,
  postal_code text,
  company_description text,
  company_website text,
  created_at timestamptz default now(),
  updated_at timestamptz default now(),
  constraint profiles_id_fkey foreign key (id) references auth.users (id)
);

create table public.products (
  id uuid primary key default gen_random_uuid(),
  sku text,
  title text,
  description text,
  unit_price numeric,
  source_url text,
  created_at timestamptz default now()
);

create table public.devis (
  id uuid primary key default gen_random_uuid(),
  user_id uuid references public.profiles (id),
  client_id uuid references public.clients (id),
  status text default 'draft',
  raw_text text,
  metadata jsonb,
  total numeric,
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

create table public.devis_items (
  id uuid primary key default gen_random_uuid(),
  devis_id uuid references public.devis (id),
  product_id uuid references public.products (id),
  description text,
  qty numeric,
  unit_price numeric,
  total numeric
);

create table public.factures (
  id uuid primary key default gen_random_uuid(),
  user_id uuid references public.profiles (id),
  client_id uuid references public.clients (id),
  raw_text text,
  metadata jsonb,
  total numeric,
  created_at timestamptz default now(),
  devis_id uuid references public.devis (id)
);

create table public.facture_items (
  id uuid primary key default gen_random_uuid(),
  facture_id uuid references public.factures (id),
  product_id uuid references public.products (id),
  description text,
  qty integer not null,
  unit_price numeric not null,
  total numeric not null
);

create table public.embeddings (
  id uuid primary key default gen_random_uuid(),
  source_table text,
  source_id uuid,
  embedding vector,
  text_excerpt text,
  created_at timestamptz default now()
);

create table public.notifications (
  id uuid primary key default gen_random_uuid(),
  user_id uuid references public.profiles (id) on delete cascade,
  title text not null,
  description text,
  type text not null default 'info',
  action_url text,
  data jsonb default '{}'::jsonb,
  read_at timestamptz,
  created_at timestamptz default now()
);

create table public.prompts (
  id serial primary key,
  name text,
  role text,
  content text,
  tags text[],
  created_at timestamptz default now()
);
