# Specify the base image
FROM python:3.8

# Update the package manager and install a simple module. The RUN command
# will execute a command on the container and then save a snapshot of the
# results. The last of these snapshots will be the final image
RUN apt-get update -y && apt-get install -y zip

# Install additional Python packages
RUN pip install jupyter 
# Install numerical Python packages
RUN pip install numpy==1.23 pandas scikit-learn 
# Install NLP Python packages
RUN pip install gensim nltk pyemd sacrebleu unbabel-comet git+https://github.com/google-research/bleurt.git \
                evaluate datasets transformers happytransformer sentence_transformers nlg-metricverse 
# Download nltk stopwords
RUN python -m nltk.downloader stopwords
# Instal Neural Networks Python packages
RUN pip install tensorflow==2.11 keras torch==1.13.0 torchmetrics
RUN pip install -q torch-scatter -f https://data.pyg.org/whl/torch-1.13.0+cpu.html
RUN pip install -q torch-sparse -f https://data.pyg.org/whl/torch-1.13.0+cpu.html
RUN pip install -q torch-geometric -f https://data.pyg.org/whl/torch-1.13.0+cpu.html
# Instal Visualization Python packages
RUN pip install matplotlib==3.5.3 rise==5.7.1 seaborn wordcloud
# Instal other tools Python packages
RUN pip install tqdm

##### New packages for TM project
RUN pip install bertopic bertopic[flair] bertopic[gensim] bertopic[spacy] bertopic[use]

# Make sure the contents of our repo are in /app
COPY . /app

# Specify working directory
WORKDIR /app/notebooks

# Use CMD to specify the starting command
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", \
     "--ip=0.0.0.0", "--allow-root"]
