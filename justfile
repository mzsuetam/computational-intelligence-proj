fetch-potsdam:
    curl https://seafile.projekt.uni-hannover.de/seafhttp/files/bc000257-9705-4fe6-bb7e-2121b75ae8ac/Potsdam.zip --output datasets/Potsdam.zip && \
    cd datasets && unzip Potsdam.zip && rm Potsdam.zip && cd Potsdam && \
    && unzip 5_Labels_for_participants.zip \
    && unzip 2_Ortho_RGB.zip \
    && cd ../..

fetch-hil-dataset:
    kaggle datasets download humansintheloop/semantic-segmentation-of-aerial-imagery && \
    mv semantic-segmentation-of-aerial-imagery.zip datasets/semantic-segmentation-of-aerial-imagery.zip && \
    cd datasets && unzip semantic-segmentation-of-aerial-imagery.zip && \
    rm semantic-segmentation-of-aerial-imagery.zip && cd ..

sync-to-remote:
    rsync -Paz --exclude-from .rsyncignore . pc-local:~/Documents/Uni/computational-intelligence/proj

sync-from-remote:
    rsync -Paz --exclude-from .rsyncignore pc-local:~/Documents/Uni/computational-intelligence/proj/ .

