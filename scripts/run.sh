#!/bin/bash
parallel sh -x ::: dann.sh deepall.sh
parallel sh -x ::: dann_mixup0.7expRatio0.5.sh deepall_mixup.sh
