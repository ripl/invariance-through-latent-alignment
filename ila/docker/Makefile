default: build release
build:
	docker build -t ila .
clean-build:
	docker build --no-cache=true -t ila .
run:
	docker run ila
# release:
# 	docker tag ila takumaynd/ila
# 	docker push takumaynd/ila
