list=$(ls *.jpg)
for img in $list; do
magick $img -resize 70% $img
done