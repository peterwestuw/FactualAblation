set TaskId=%1

REM .\Python37\python.exe -m spacy download en_core_web_sm
.\Python37\Scripts\pip.exe install -r .\data_processing\requirements.txt
.\Python37\python.exe -c "import nltk; nltk.download('punkt'); nltk.download('words');  nltk.download('stopwords');"

.\Python37\python.exe .\data_processing\post_process_2.py --input-path \\nlp-storage\scratch\users\t-pewest\WikiCommentEdit\out\Peter-08-20\ --output-path \\nlp-storage\scratch\users\t-pewest\WikiCommentEdit\out\%SNAPSHOT_FILENAME%\ --index %TaskId%