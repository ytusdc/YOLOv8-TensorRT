current_dir=$PWD
cd ${current_dir}
echo ${current_dir}

build_dir="${current_dir}/build"
echo ${build_dir}

if [ ! -d "$build_dir" ]; then
  mkdir -p "$build_dir"
fi
cd ${build_dir}

rm -rf *
cmake ..
make