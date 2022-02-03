import os
import sys
import shutil
import json
from dotenv import load_dotenv
load_dotenv()

if not os.path.exists('websites'): os.mkdir('websites')

repo_path = os.getenv('REPO_DIR')

#This is up here because we need to pull from the pipeline settings in the networkplot data folder 
#TODO: Where to put pipeline settings or how to ensure that the settings are the same between all included plots?
topic_network_folder = os.path.join(repo_path, 'visualization/topic_network')
with open(os.path.join(topic_network_folder, 'data','pipeline_settings.json'), 'r') as f:
   pipeline_settings = json.load(f)

regex = pipeline_settings['regex']

website_name = regex.replace(' ', '_')
website_name = "".join(x for x in website_name if x.isalnum() or x == '_') #Some disallowed things from a regex in folder/filenames

website_dir = 'websites/{}'.format(website_name)
if not os.path.exists(website_dir):
    os.mkdir(website_dir)
    os.mkdir(os.path.join(website_dir, website_name))

shutil.copyfile('main.css', os.path.join(website_dir, website_name, 'main.css'))

shutil.copyfile(
    os.path.join(topic_network_folder, 'output', 'topic_network.html'),
    os.path.join(website_dir, website_name, 'topic_network.html'),
)


img_dir = os.path.join(website_dir, website_name, 'img')
if not os.path.exists(img_dir):
    os.mkdir(img_dir)

for fn in os.listdir('img'):
    shutil.copyfile(os.path.join('img',fn), os.path.join(img_dir,fn))

## search term percentage plot
corpus_analysis_folder = os.path.join(repo_path, 'text_data/semantic/analysis')


fns = ['full_corpus_pub_annual.png', 'search_term_pub_percent.png']
for fn in fns:
    shutil.copyfile(
        os.path.join(corpus_analysis_folder, 'output', fn),
        os.path.join(img_dir, fn),
    )

## topic trends plots
topic_trends_folder = os.path.join(repo_path, 'visualization/topic_trends')

fns = ['top_slopes_plot.png', 'neg_slopes_plot.png']
for fn in fns:
    shutil.copyfile(
        os.path.join(topic_trends_folder, 'output', fn),
        os.path.join(img_dir, fn),
    )



from jinja2 import Template

# Our template. Could just as easily be stored in a separate file
with open('template.html', 'r') as file_:
    t = Template(file_.read())

print(pipeline_settings)
render_str = t.render(pipeline_settings)

output_path = os.path.join(website_dir, '{}.html'.format(website_name))
with open(output_path, 'w') as f:
    f.write(render_str)