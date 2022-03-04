#!/bin/bash

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} "$@" || curl -G "https://smsapi.free-mobile.fr/sendmsg?user=18146642&pass=T14IZeKeJgBJsB&msg=Ton job $(echo "$@") a planté sur la machine $(hostname), gpu ${CUDA_VISIBLE_DEVICES}, bisous."

