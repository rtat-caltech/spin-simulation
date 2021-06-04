# syntax=docker/dockerfile:1

FROM jupyter/datascience-notebook

ARG WDIR="/home/${NB_USER}/spin-simulation"

WORKDIR "${WDIR}"

COPY install.jl install.jl

RUN julia install.jl

COPY examples examples

USER root
RUN fix-permissions "${WDIR}/examples"
USER $NB_UID
