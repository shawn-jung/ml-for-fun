# delete previous requirements files if exist
rm -f requirements.txt requirements-dev.txt

# for app serving
uv export --format requirements-txt --no-annotate --no-hashes -o requirements.txt
uv export --format requirements-txt --no-annotate --no-hashes --extra dev -o requirements-dev.txt
