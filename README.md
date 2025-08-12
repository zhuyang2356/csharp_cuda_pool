# csharp_cuda_pool
高性能计算的时候，将CUDA程序包装成DLL再让C#,JAVA调用。
本项目旨在提供一个框架，在C#计算的时候，只要用CMAKE生成DLL，C#里先调用DLL，初始化好显存池，然后再调用方法即可。省去申请显存时间。
