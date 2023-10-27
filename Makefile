
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

setup_env:
	@echo "Setting up environment..."
	pip install pip==21.2.4
	pip install -r requirements.txt
	@echo "Building fairseq..."
	cd fairseq && pip install -e .
	@echo "Done."

.PHONY: download_wsdm_vqa_dataset
download_wsdm_vqa_dataset:
	cd ./datasets; \
	dataset_source=https://zenodo.org/records/7570356/files/; \
	splits="train_sample test_public"; \
	@echo "Downloading dataset..."; \
	for split in $$splits; do \
		wget $${dataset_source}$$split.zip?download=1 -O ./$$split.zip && unzip ./$$split.zip -d ./$$split & \
	done; \
	wait; \
	@echo "Done."
