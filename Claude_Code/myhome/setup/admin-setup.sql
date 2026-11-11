-- =====================================================================
-- 記事管理ページ（admin.html）用の初回セットアップ
-- Supabase の「SQL Editor」に全部貼り付けて「Run」を押してください。
-- 何度実行しても大丈夫です。
--
-- 実行する前に：
--   Supabase の Authentication → Users → 「Add user」→「Create new user」で
--   自分用のユーザーを作ってください（「Auto Confirm User」をオン）。
--   そして一番下の 'YOUR_EMAIL@example.com' をそのメールアドレスに書き換えてください。
-- =====================================================================

-- 1. 管理者リスト（ここに登録された人だけが記事を書き込めます）
create table if not exists public.admins (
    user_id uuid primary key references auth.users(id) on delete cascade
);
alter table public.admins enable row level security;
-- admins にはポリシーを作らない＝サイトからは読み書きできない

-- 2. ログイン中の人が管理者かどうかを返す関数
create or replace function public.is_admin()
returns boolean
language sql
stable
security definer
set search_path = public
as $$
    select exists (select 1 from public.admins where user_id = auth.uid());
$$;
grant execute on function public.is_admin() to anon, authenticated;

-- 3. posts 表：管理者だけが追加・更新・削除できる（読むのは今まで通り誰でも可）
drop policy if exists "admin insert posts" on public.posts;
drop policy if exists "admin update posts" on public.posts;
drop policy if exists "admin delete posts" on public.posts;

create policy "admin insert posts" on public.posts
    for insert to authenticated with check (public.is_admin());
create policy "admin update posts" on public.posts
    for update to authenticated using (public.is_admin()) with check (public.is_admin());
create policy "admin delete posts" on public.posts
    for delete to authenticated using (public.is_admin());

-- 4. 画像（Storage の 0505 バケット）：管理者だけがアップロード・削除できる
drop policy if exists "admin select 0505" on storage.objects;
drop policy if exists "admin insert 0505" on storage.objects;
drop policy if exists "admin delete 0505" on storage.objects;

create policy "admin select 0505" on storage.objects
    for select to authenticated using (bucket_id = '0505' and public.is_admin());
create policy "admin insert 0505" on storage.objects
    for insert to authenticated with check (bucket_id = '0505' and public.is_admin());
create policy "admin delete 0505" on storage.objects
    for delete to authenticated using (bucket_id = '0505' and public.is_admin());

-- 5. projects 表：管理者だけが追加・更新・削除できる
--    （管理ページはログインした状態で読むので、管理者用の読み取り設定も追加）
drop policy if exists "admin select projects" on public.projects;
drop policy if exists "admin insert projects" on public.projects;
drop policy if exists "admin update projects" on public.projects;
drop policy if exists "admin delete projects" on public.projects;

create policy "admin select projects" on public.projects
    for select to authenticated using (public.is_admin());
create policy "admin insert projects" on public.projects
    for insert to authenticated with check (public.is_admin());
create policy "admin update projects" on public.projects
    for update to authenticated using (public.is_admin()) with check (public.is_admin());
create policy "admin delete projects" on public.projects
    for delete to authenticated using (public.is_admin());

-- 6. 自分を管理者に登録（メールアドレスを書き換えてください）
insert into public.admins (user_id)
select id from auth.users where email = 'YOUR_EMAIL@example.com'
on conflict do nothing;

-- 確認：下の結果に自分のメールアドレスが出れば成功です
select u.email from public.admins a join auth.users u on u.id = a.user_id;
