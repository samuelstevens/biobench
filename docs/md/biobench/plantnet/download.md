Module biobench.plantnet.download
=================================

Functions
---------

`main(args: biobench.plantnet.download.Args)`
:   

Classes
-------

`Args(dir: str = '.', chunk_size_kb: int = 1, download: bool = True, unzip: bool = True)`
:   Args(dir: str = '.', chunk_size_kb: int = 1, download: bool = True, unzip: bool = True)

    ### Class variables

    `chunk_size_kb: int`
    :   how many KB to download at a time before writing to file.

    `dir: str`
    :   where to save data.

    `download: bool`
    :   whether to download images [29.5GB].

    `unzip: bool`
    :   whether to unzip images.