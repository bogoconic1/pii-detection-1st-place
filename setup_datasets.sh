hdir=$(pwd)

mkdir models
mkdir data
cd data

kaggle datasets download -d ympaik/pii-1st-solution-datasets
unzip pii-1st-solution-datasets.zip -d ./
rm pii-1st-solution-datasets.zip

kaggle datasets download -d nbroad/pii-dd-mistral-generated
unzip pii-dd-mistral-generated.zip -d pii-dd-mistral-generated
rm pii-dd-mistral-generated

kaggle datasets download -d mpware/pii-mixtral8x7b-generated-essays
unzip pii-mixtral8x7b-generated-essays.zip -d pii-mixtral8x7b-generated-essays
rm pii-mixtral8x7b-generated-essays.zip

cd $hdir
