const SUPABASE_URL = 'https://fqzkososytnhtgiwzygx.supabase.co';
const SUPABASE_KEY = 'sb_publishable_EQK-g-JxrKvxuv7cHh0lUw_e49UicoR';

// 公開ページ用：管理ページのログイン状態を使わず、常に「ゲスト」として読み込む。
// （ログイン状態を共有すると、ゲスト向けの読み取り設定しかない表が見えなくなるため）
const db = supabase.createClient(SUPABASE_URL, SUPABASE_KEY, {
    auth: { persistSession: false, autoRefreshToken: false, detectSessionInUrl: false, storageKey: 'myhome-public' }
});

function formatDate(dateStr) {
    const d = new Date(dateStr);
    return `${d.getFullYear()}年${d.getMonth() + 1}月${d.getDate()}日`;
}

function imageUrl(filename) {
    return `${SUPABASE_URL}/storage/v1/object/public/0505/${encodeURIComponent(filename)}`;
}
