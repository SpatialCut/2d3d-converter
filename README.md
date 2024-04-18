# ECR image path

055843083853.dkr.ecr.us-west-1.amazonaws.com/fetecr

# Local image path
my-sagemaker-model

# ECR auth
aws ecr get-login-password --region us-west-1 | docker login --username AWS --password-stdin 055843083853.dkr.ecr.us-west-1.amazonaws.com




docker build -t my-sagemaker-model .

docker tag my-sagemaker-model 055843083853.dkr.ecr.us-west-1.amazonaws.com/fetecr
docker push 055843083853.dkr.ecr.us-west-1.amazonaws.com/fetecr

docker run -it --rm --entrypoint /bin/bash my-sagemaker-model

docker run -it --rm my-sagemaker-model

docker run -it --rm  --entrypoint /bin/bash 055843083853.dkr.ecr.us-west-1.amazonaws.com/fetecr


docker system prune -f
