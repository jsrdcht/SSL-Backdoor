#!/bin/bash
# 验证所有文件的中文注释是否已翻译为英文

echo "=========================================="
echo "检查剩余的中文注释..."
echo "=========================================="

FILES=(
    "tools/run_clip_backdoor.py"
    "tools/run_clip_backdoor.sh"
    "tools/train_badclip.sh"
    "tools/train_clip.py"
    "tools/train_clip.sh"
    "configs/clip/badclip/eval_zeroshot_imagenet.yaml"
    "configs/clip/badclip/optimize_trigger_banana.yaml"
    "configs/clip/badclip/poison_badclip_banana_cc3m.yaml"
    "configs/clip/badclip/train_badclip_banana_cc3m.yaml"
    "configs/clip/clip_backdoor/clip_vit_b16_cc3m_poisoned.yaml"
    "configs/clip/clip_backdoor/eval_zeroshot_imagenet.yaml"
    "configs/clip/clip_backdoor/poison_sslbkd_banana_cc3m.yaml"
)

has_chinese=false

for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        chinese_lines=$(grep -n "[一-鿿]" "$file" 2>/dev/null || true)
        if [ -n "$chinese_lines" ]; then
            echo "❌ $file 仍有中文："
            echo "$chinese_lines"
            echo ""
            has_chinese=true
        fi
    else
        echo "⚠️  文件不存在: $file"
    fi
done

if [ "$has_chinese" = false ]; then
    echo "✅ 所有文件的中文注释已翻译完成！"
    exit 0
else
    echo "❌ 仍有文件包含中文注释"
    exit 1
fi
