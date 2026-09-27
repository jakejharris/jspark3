FROM ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks@sha256:447114ee77d14c9b4732ee23978ada2a0ee9027868a231d6fd42700a8b25be1d AS instanttensor
FROM ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks@sha256:9bb1557a4234fce63d59599e44d10747eabd742beb337eebf9e7070be8a0fd58
COPY --from=instanttensor /usr/local/lib/python3.12/dist-packages/instanttensor /usr/local/lib/python3.12/dist-packages/instanttensor
COPY --from=instanttensor /usr/local/lib/python3.12/dist-packages/instanttensor-0.2.0.dist-info /usr/local/lib/python3.12/dist-packages/instanttensor-0.2.0.dist-info
COPY instanttensor-files.json /opt/jspark3-v13/instanttensor-files.json
RUN python3 -S - <<'PY'
import hashlib,json,pathlib
root=pathlib.Path('/usr/local/lib/python3.12/dist-packages')
manifest=json.loads(pathlib.Path('/opt/jspark3-v13/instanttensor-files.json').read_text())
for name,digest in manifest.items():
 assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
print('JSPARK3_INSTANTTENSOR_PACKAGE_PASS')
PY
ENV ABLIT=0
LABEL org.opencontainers.image.title="JSpark3 Cadence v1.3.0 stock derivative" \
      jspark3.release="v1.3.0" \
      jspark3.stock="1" \
      jspark3.source.base="bde15cf769203cf9c387720810f0cc09dea55c66" \
      jspark3.instanttensor="0.2.0"
