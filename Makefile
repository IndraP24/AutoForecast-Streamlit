# Arcane incantation to print all the other targets, from https://stackoverflow.com/a/26339924
help:
	@$(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST)) : 2>/dev/null | awk -v RS= -F: '/^# File/,/^# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}' | sort | egrep -v -e '^[^[:alnum:]]' -e '^$@$$'

# Create conda environment with required dependencies
conda-update:
	conda env update --prune -f environment.yml

# Compile and install exact pip packages
pip-tools:
	pip install pip-tools pystan==2.19.1.1
	pip-compile requirements.in
	pip-sync requirements.txt