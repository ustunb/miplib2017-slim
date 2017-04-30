#!/bin/bash
# This script will create all of the SLIM IP instances
# It will then produce the zip files for the submission to MIPLIB 2017
# You should run this script from the directory where it is contained

all_data_names=("breastcancer" "mushroom" "adult")
all_problem_types=("best" "max_5_features" "regularized")

#set directories
repo_dir=$(pwd)
instances_dir="${repo_dir}/instances"
models_dir="${repo_dir}/models"
data_dir="${models_dir}/data"
misc_dir="${repo_dir}/misc"

#create MPS files for each dataset and problem type
for data_name in ${all_data_names[*]}; do
for problem_type in ${all_problem_types[*]}; do

data_file="${data_dir}/${data_name}_processed.csv"
instance_file="${instances_dir}/${data_name}_${problem_type}.mps"
instance_info="${misc_dir}/${data_name}_${problem_type}.p"

max_coef=10
max_offset=100

case ${problem_type} in
    best)
        max_size=-1
        c0_value=-1
    ;;
    max_5_features)
        max_size=5
        c0_value=-1
    ;;
    regularized)
       max_size=-1
       c0_value=0.01
    ;;
esac

python "${models_dir}/create_slim_instance.py"  \
    --data_file "${data_file}" \
    --instance_file "${instance_file}" \
    --instance_info "${instance_info}" \
    --c0_value "${c0_value}" \
    --max_size "${max_size}" \
    --max_coef "${max_coef}" \
    --max_offset "${max_offset}"

done
done

zip -r "${repo_dir}/slim-models.zip" "${models_dir}"
zip -r "${repo_dir}/slim-instances.zip" "${instances_dir}"
zip -r "${repo_dir}/slim-misc.zip" "${misc_dir}"

exit