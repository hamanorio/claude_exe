const SUPABASE_URL = 'https://fqzkososytnhtgiwzygx.supabase.co';
const SUPABASE_KEY = 'sb_publishable_EQK-g-JxrKvxuv7cHh0lUw_e49UicoR';

const db = supabase.createClient(SUPABASE_URL, SUPABASE_KEY);

function formatDate(dateStr) {
    const d = new Date(dateStr);
    return `${d.getFullYear()}年${d.getMonth() + 1}月${d.getDate()}日`;
}

function imageUrl(filename) {
    return `${SUPABASE_URL}/storage/v1/object/public/0505/${encodeURIComponent(filename)}`;
}
