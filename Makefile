docker_build:
	docker build . --tag wsdm_vqa --network host

docker_run:
	docker run \
		-it \
		--rm \
		--gpus all \
		--network host \
		--entrypoint '/bin/bash' \
		-v ${PWD}:/wsdm \
		xexanoth/rep:fix20230110-2

docker_rep:
	docker run \
		-it \
		--rm \
		--gpus all \
		--network host \
		--entrypoint '/bin/bash' \
		-v /home/P76104419/ICCV/dataset:/wsdm/dataset \
		-v /home/P76104419/ICCV/dataset/rep_data:/mnt/data \
		-v /home/P76104419/ICCV/dataset/rep_output:/mnt/output \
		xexanoth/rep:fix20230110-2

train:
	bash ./scripts/train_single_exp.sh