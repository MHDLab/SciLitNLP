mkdir temp_gz
wget https://s3-us-west-2.amazonaws.com/ai2-s2-research-public/open-corpus/2022-01-01/manifest.txt -O manifest.txt

FILENAME="manifest.txt"
LINES=$(cat $FILENAME)

#https://codefather.tech/blog/bash-loop-through-lines-file/
COUNTER=0
MAX_FILES=5
for LINE in $LINES
do
    if [ $COUNTER -eq $MAX_FILES ]; then
        break
    fi

    url="https://s3-us-west-2.amazonaws.com/ai2-s2-research-public/open-corpus/2022-01-01/$LINE"
    out="temp_gz/$LINE"

    wget $url -O $out

    COUNTER=$((COUNTER+1))
done