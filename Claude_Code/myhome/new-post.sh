#!/bin/bash

# ブログ記事作成スクリプト
# 使い方: ./new-post.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BLOG_DIR="$SCRIPT_DIR/blog"
TEMPLATE="$BLOG_DIR/_template.html"
INDEX_FILE="$SCRIPT_DIR/index.html"
BLOG_LIST_FILE="$SCRIPT_DIR/blog.html"

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

# 概要入力
read -p "記事の概要を入力してください: " SUMMARY

# 今日の日付
TODAY=$(date "+%Y年%-m月%-d日")

# ファイル名
FILENAME="post${NEXT_NUM}.html"
FILEPATH="$BLOG_DIR/$FILENAME"

# テンプレートをコピーして置換
cp "$TEMPLATE" "$FILEPATH"
sed -i '' "s/【タイトル】/$TITLE/g" "$FILEPATH"
sed -i '' "s/【日付】/$TODAY/g" "$FILEPATH"

echo ""
echo "記事ファイルを作成しました: blog/$FILENAME"

# index.html の最新記事を更新
NEW_LATEST="                <!-- LATEST_POST_START -->\n                <article class=\"post-card\">\n                    <div class=\"post-date\">$TODAY<\/div>\n                    <h3><a href=\"blog\/$FILENAME\">$TITLE<\/a><\/h3>\n                    <p>$SUMMARY<\/p>\n                    <a href=\"blog\/$FILENAME\" class=\"read-more\">続きを読む →<\/a>\n                <\/article>\n                <!-- LATEST_POST_END -->"

# index.html を更新
sed -i '' '/<!-- LATEST_POST_START -->/,/<!-- LATEST_POST_END -->/c\
'"$NEW_LATEST"'' "$INDEX_FILE"

echo "index.html の最新記事を更新しました"

# blog.html に新しい記事を追加
NEW_BLOG_ITEM="            <!-- BLOG_LIST_START -->\n            <article class=\"blog-item\">\n                <div class=\"blog-date\">$TODAY<\/div>\n                <h2><a href=\"blog\/$FILENAME\">$TITLE<\/a><\/h2>\n                <p>$SUMMARY<\/p>\n                <a href=\"blog\/$FILENAME\" class=\"read-more\">続きを読む →<\/a>\n            <\/article>\n            <!-- BLOG_LIST_NEW -->"

sed -i '' 's/<!-- BLOG_LIST_START -->/'"$NEW_BLOG_ITEM"'/' "$BLOG_LIST_FILE"

echo "blog.html に記事を追加しました"

echo ""
echo "完了！ blog/$FILENAME を編集して記事の内容を書いてください。"
