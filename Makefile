requirements:
	poetry export --without-hashes --format=requirements.txt > requirements.txt


docs:
	pdoc -o ./docs/html scibert -d google
