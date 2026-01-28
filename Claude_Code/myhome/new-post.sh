#!/bin/bash

# ブログ記事作成スクリプト
# 使い方: ./new-post.sh

BLOG_DIR="$(dirname "$0")/blog"
TEMPLATE="$BLOG_DIR/_template.html"

# 次の記事番号を自動取得
LAST_NUM=$(ls -1 "$BLOG_DIR"/post*.html 2>/dev/null | grep -o 'post[0-9]*' | sed 's/post//' | sort -n | tail -1)
if [ -z "$LAST_NUM" ]; then
    NEXT_NUM=1
else
    NEXT_NUM=$((LAST_NUM + 1))
fi

echo "=== 新しいブログ記事を作成します ==="
echo ""

# タイトル入力
read -p "記事のタイトルを入力してください: " TITLE

# 今日の日付
TODAY=$(date "+%Y年%-m月%-d日")

# ファイル名
FILENAME="post${NEXT_NUM}.html"
FILEPATH="$BLOG_DIR/$FILENAME"

# テンプレートをコピーして置換
cp "$TEMPLATE" "$FILEPATH"
sed -i '' "s/【タイトル】/$TITLE/g" "$FILEPATH"
sed -i '' "s/【日付：例 2026年1月28日】/$TODAY/g" "$FILEPATH"

echo ""
echo "記事ファイルを作成しました: blog/$FILENAME"
echo ""
echo "次のステップ:"
echo "1. blog/$FILENAME を編集して記事の内容を書く"
echo "2. blog.html に以下を追加して記事をリストに載せる:"
echo ""
echo "            <article class=\"blog-item\">"
echo "                <div class=\"blog-date\">$TODAY</div>"
echo "                <h2><a href=\"blog/$FILENAME\">$TITLE</a></h2>"
echo "                <p>記事の概要をここに書く</p>"
echo "                <a href=\"blog/$FILENAME\" class=\"read-more\">続きを読む →</a>"
echo "            </article>"
echo ""
