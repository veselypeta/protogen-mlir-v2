#!/bin/bash

# This is a script used to generate JSON Validation schema files
# these are used within tests to check that the generated json conforms
# to the schema

if which npm > /dev/null
then
  package="typescript-json-schema"
  npm list -g | grep $package || npm install -g $package

  echo "-- generating json schema files"

  ts_file="murphi_json_schema.ts"
  types=("TypeDecl" "ConstDecl" "MurphiType" "Murphi_json_schema" "TypeDescription") # add types to be generated here

  for t in "${types[@]}"; do
    $package $ts_file "$t" --required --titles > "generated_json_schemas/gen_${t}.json"
  done

  else
      echo "NPM is not installed, please install and re-run the script"
      exit 1
fi

exit 0