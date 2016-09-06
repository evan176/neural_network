#!/bin/bash


function create_env {
  if [[ "${1}" == "py26" ]]; then
    virtualenv "local" --python=python2.6 --never-download
  elif [[ "${1}" == "py27" ]]; then
    virtualenv "local" --python=python2.7 --never-download
  elif [[ "${1}" == "py32" ]]; then
    virtualenv "local" --python=python3.2 --never-download
  elif [[ "${1}" == "py33" ]]; then
    virtualenv "local" --python=python3.3 --never-download
  elif [[ "${1}" == "py34" ]]; then
    virtualenv "local" --python=python3.4 --never-download
  elif [[ "${1}" == "py35" ]]; then
    virtualenv "local" --python=python3.5 --never-download
  else
    virtualenv "local" --python=python2.7 --never-download
  fi
}


function package_install {
  local/bin/pip install -r requirements.txt
}


function project_test {
  . local/bin/activate
  python -m pytest tests
}


function project_clean {
  deactivate
  rm -rf local
  rm neural_network/*.pyc
  rm tests/*.pyc
}


# Command Line Arguments
if [[ "${#}" -eq 0 ]]; then
  HELP="1"
else
  while [[ "${#}" > 0 ]]; do
    key="${1}"
    case "${key}" in
      -p|--python)
        PYTHON="${2}"
        shift
        ;;
      -i|--install)
        INSTALL="1"
        shift
        ;;
      -t|--test)
        TEST="1"
        shift
        ;;
      -c|--clean)
        CLEAN="1"
        shift
        ;;
      *)
        ;;
  esac
    shift
  done
fi


if [[ -n "${HELP}" ]]; then
  echo 1
else
  if [[ -n "${PYTHON}" ]]; then
    create_env "${PYTHON}"
  fi
  if [[ -n "${INSTALL}" ]]; then
    package_install
  fi
  if [[ -n "${TEST}" ]]; then
    project_test
  fi
  if [[ -n "${CLEAN}" ]]; then
    project_clean
  fi
fi
