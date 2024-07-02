# ECR image path

055843083853.dkr.ecr.us-west-1.amazonaws.com/fetecr


# Image built and push

- docker build -t my-sagemaker-model .

- docker tag my-sagemaker-model 055843083853.dkr.ecr.us-west-1.amazonaws.com/fetecr

- aws ecr get-login-password --region us-west-1 | docker login --username AWS --password-stdin 055843083853.dkr.ecr.us-west-1.amazonaws.com

- docker push 055843083853.dkr.ecr.us-west-1.amazonaws.com/fetecr


# Run image locally

- docker run -it --rm --entrypoint /bin/bash my-sagemaker-model

- docker run -it --rm my-sagemaker-model

- docker run -it --rm  --entrypoint /bin/bash 055843083853.dkr.ecr.us-west-1.amazonaws.com/fetecr

- docker system prune -f

- docker cp ForBiggerMeltdowns.mp4 1793505e8008:/app/ForBiggerMeltdowns.mp4



# SageMaker endpoint set up

## create SageMaker endpoint configuration
- name: 2d3dgpumlp3-2 (or anything else)
- variant: 
  - model: 2d3dsagemodel
  - Instance type: ml.g4dn.xlarge

## create SageMaker endpoint

### create from console

- name: 2d3dmlg4
- endpoint comfiguratin: as the one above

### use aws cli

- aws sagemaker list-endpoints
- aws sagemaker create-endpoint --endpoint-name 2d3dmlg4 --endpoint-config-name 2d3dgpumlp3-2
- aws sagemaker delete-endpoint --endpoint-name 2d3dmlg4


### EC2

- scp -i "/Users/feiwu/Downloads/oldmodelssh.pem" process_video.py requirements.txt ec2-user@13.52.75.190:~
- 

### NEW: on ec2

- docker build -t my_tf_app .

- docker run -d --rm -p 80:8080   -e AWS_ACCESS_KEY_ID=   -e AWS_SECRET_ACCESS_KEY=  --gpus all   my_tf_app