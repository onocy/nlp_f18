import nltk
import csv
import re

# NLTK complains and fails to tokenize without this. May be different on your machine.
# nltk.download('punkt')

artist_lyrics = {}
with open('songdata.csv') as file:
	file_csv = csv.reader(file, quotechar='"')
	i = 0
	for row in file_csv:
		artist = row[0]
		lyrics = nltk.word_tokenize(re.sub('\'|,|\(|\)|\?|\!', '', row[3].lower()))
		if artist not in artist_lyrics:
			artist_lyrics[artist] = []
		artist_lyrics[artist] += lyrics

print(type(artist_lyrics['ABBA']))
print(artist_lyrics['ABBA'])