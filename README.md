# ECR image path

055843083853.dkr.ecr.us-west-1.amazonaws.com/fetecr

# Local image path
my-lambda-function

# ECR auth
aws ecr get-login-password --region us-west-1 | docker login --username AWS --password-stdin 055843083853.dkr.ecr.us-west-1.amazonaws.com




docker build -t my-lambda-function .

docker tag my-lambda-function 055843083853.dkr.ecr.us-west-1.amazonaws.com/fetecr
docker push 055843083853.dkr.ecr.us-west-1.amazonaws.com/fetecr

docker run -it --rm --entrypoint /bin/bash my-lambda-function

docker run -it --rm  --entrypoint /bin/bash 055843083853.dkr.ecr.us-west-1.amazonaws.com/fetecr


docker system prune -f
