Module biobench.newt.download
=============================

Functions
---------

`main(args: biobench.newt.download.Args)`
:   

Classes
-------

`Args(dir: str = '.', chunk_size_kb: int = 1, images: bool = True, labels: bool = True)`
:   Args(dir: str = '.', chunk_size_kb: int = 1, images: bool = True, labels: bool = True)

    ### Class variables

    `chunk_size_kb: int`
    :   how many KB to download at a time before writing to file.

    `dir: str`
    :   where to save data.

    `images: bool`
    :   whether to download images [4.1GB].

    `labels: bool`
    :   whether to download labels.