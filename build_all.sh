
dependencies_path=$(pwd)/dependencies

if [ ! -d "$dependencies_path/boost/lib" ]; then
  echo "[Boost] has not been found"
  rm -rf ./build
  cmake -Bbuild -DCMAKE_BUILD_TYPE=Release && cmake --build build --config Release --target external_boost
else
  echo "[Boost] found skip this stage"
fi

if [ ! -d "$dependencies_path/bayesopt/lib" ]; then
  echo "[Bayesopt] has not been found"
  rm -rf ./build
  cmake -Bbuild -DCMAKE_BUILD_TYPE=Release && cmake --build build --config Release --target bayesopt_external
else
  echo "[Bayesopt] found skip this stage"
fi

cmake -Bbuild -DCMAKE_BUILD_TYPE=Release && cmake --build build --config Release