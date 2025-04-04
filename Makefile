#!/usr/bin/make

update_env_file:
	cat ./env.example > ./.env

build_base_image:
	docker build -t s_trans_base_image -f ./deploy/Dockerfile_base .


build_torch_image:
	docker build -t s_trans_torch_image -f ./deploy/Dockerfile_torch .


build_frontend_image:
	docker build -t s_trans_frontend_image -f ./deploy/Dockerfile_frontend .


build_backend_image:
	docker build -t s_trans_backend_image -f ./deploy/Dockerfile_backend .


build_llm_image:
	docker build -t s_trans_llm_image -f ./deploy/Dockerfile_llm .


drop_images:
	docker rmi frontend s_trans_frontend_image backend s_trans_backend_image llm_inference s_trans_llm_image  s_trans_torch_image s_trans_base_image


clean:
	docker system prune -f
	docker volume prune -f
	docker network prune -f
	docker image prune -f
	docker container prune -f


build_images:
	make update_env_file
	make build_base_image
	make build_torch_image
	make build_llm_image
	make build_backend_image
	make build_frontend_image


up:
	docker compose -f ./docker-compose.yml up

down:
	docker compose -f ./docker-compose.yml down


full_stop:
	make update_env_file && make down && make drop_images && make clean


full_start:
	make update_env_file && make build_images && make up


full_restart:
	make full_stop
	make full_start
