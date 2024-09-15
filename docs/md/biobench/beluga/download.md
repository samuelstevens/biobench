Module biobench.beluga.download
===============================

Functions
---------

`main(args: biobench.beluga.download.Args)`
:   

Classes
-------

`Args(dir: str = '.', chunk_size_kb: int = 1, download: bool = True, expand: bool = True)`
:   Args(dir: str = '.', chunk_size_kb: int = 1, download: bool = True, expand: bool = True)

    ### Class variables

    `chunk_size_kb: int`
    :   how many KB to download at a time before writing to file.

    `dir: str`
    :   where to save data.

    `download: bool`
    :   whether to download images.

    `expand: bool`
    :   whether to expand tarfiles into a folder.