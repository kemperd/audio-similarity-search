# audio-similarity-search

Tested with Python 3.13.2

1. git clone https://github.com/kemperd/audio-similarity-search
2. cd audio-similarity-search
3. wget https://github.com/karoldvl/ESC-50/archive/master.zip
4. unzip master.zip
5. conda create env -n audio-similarity-search
6. pip install -r requirements.txt
7. Open notebook process_esc50.ipynb and run this fully

Now you can experiment with the two audio similarity search approaches: 

1. The vector database search approach is in notebook vector_db_search.ipynb.
2. The audio classification finetuning approach is in notebook train_classfyer.ipynb
