REPO_PATH='C:\Users\aspit\Git\NLP\SciLitNLP'


#https://stackoverflow.com/questions/33950596/iterating-through-json-array-in-shell-script
#https://stackoverflow.com/questions/53967693/how-to-run-jq-from-gitbash-in-windows
# read each item in the JSON array to an item in the Bash array
readarray -t my_array < <(jq -c '.[]' all_site_info.json)

# iterate through the Bash array
#https://stackoverflow.com/questions/1335815/how-to-slice-an-array-in-bash
for item in "${my_array[@]:0:10}"; do
  regex=$(jq '.regex' <<< "$item")
  regex=$(echo $regex | xargs echo)

  long_name=$(echo $long_name | xargs echo)
  long_name=$(jq '.long_name' <<< "$item")

  #Data pipeline starts here
  cd "$REPO_PATH/text_data/semantic"
  python gen_search_index_database.py -r "$regex" #-sl 1e6

  cd "$REPO_PATH/text_data/semantic/analysis"
  python indexed_search_analysis.py -r "$regex"

  cd "$REPO_PATH/modeling/corex"
  python genmodel_soc.py -r "$regex" -tw -n 50


  cd "$REPO_PATH/visualization/topic_trends"
  python topic_trends_script.py

  cd "$REPO_PATH/visualization/topic_network"
  python prep_data_corex.py -d soc --min-edge-weight 1
  python gen_networkplot.py 

  cd "$REPO_PATH/website_gen"
  python website_gen.py -ln "$long_name"

done