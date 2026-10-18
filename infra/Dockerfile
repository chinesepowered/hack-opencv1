# NEVER BUILT OR PUSHED. See the banner in template.yaml and docs/deploy.md.
#
# Container image rather than a zip because OpenCV 5 plus numpy and
# scikit-learn comfortably exceeds the 250 MB unzipped Lambda layer limit.
# arm64 to match the Graviton architecture in template.yaml; no timing
# comparison between architectures exists for this project, because none was
# ever run.

FROM public.ecr.aws/lambda/python:3.13-arm64

# opencv-python-headless is the right build for Lambda: no GUI libraries, so
# no libGL.so.1 to install and nothing that expects an X display.
COPY infra/requirements-lambda.txt ${LAMBDA_TASK_ROOT}/requirements.txt
RUN python -m pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

COPY src/sentinel ${LAMBDA_TASK_ROOT}/sentinel
COPY infra/handler.py ${LAMBDA_TASK_ROOT}/handler.py

CMD ["handler.lambda_handler"]
