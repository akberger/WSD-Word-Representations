# WSD-Word-Representations

The aim of this project was to perform word sense disambiguation on a corpus before training Word2Vec on the corpus so that polysemous words would have different vector representations for each of their respective meanings. Word sense disambiguation was done by creating a feature vector for each word in a given word list. Feature vectors were created based on the contextual environment of words.

Usage: Pipeline.py <indir> <outdir> <word_list_file> <model_dir> <vec_files_path> <clutering='kmeans'> <iters=3>
 
indir: directory containing .txt files
outdir: directory to store sense disambiguated .txt files (I recommend writing to external memory) 
word_list_file: .txt file specifying words to perform WSD on 
vec_files_path: file to store WSD vectors in (Also recommend writing to external memory)
clustering: only kmeans works at the moment
iters: How many iterations to run the pipeline. More iterations tends to yield better performance trading off for time.
