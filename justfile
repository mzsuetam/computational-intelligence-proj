fetch-hil-dataset:
    kaggle datasets download humansintheloop/semantic-segmentation-of-aerial-imagery && \
    mv semantic-segmentation-of-aerial-imagery.zip datasets/semantic-segmentation-of-aerial-imagery.zip && \
    cd datasets && unzip semantic-segmentation-of-aerial-imagery.zip && \
    rm semantic-segmentation-of-aerial-imagery.zip && cd ..

sync-to-remote:
    rsync -Paz --exclude-from .rsyncignore . pc-local:~/Documents/Uni/computational-intelligence/proj

sync-from-remote:
    rsync -Paz --exclude-from .rsyncignore pc-local:~/Documents/Uni/computational-intelligence/proj/ .
