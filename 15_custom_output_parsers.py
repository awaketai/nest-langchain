# LangChain允许继承基类来轻松扩展其功能
# 为什么要自定义解析器：
# 1.处理非标准格式：模型可能被微调，以输出特定的、非JSON或CSV的格式
# 2.增加健壮性：在解析逻辑中加入自定义的错误处理、重试或者修正逻辑
# 3.封装复杂逻辑：将复杂的数据转换和验证逻辑封装在一个可复用的解析器中

# 如何自定义输出解析器
# 1.继承BaseOutputParser: langchain_core.output_parsers.BaseOutputParser
# 2.实现parse()方法，这是解析器的核心。这个方法接收一个来自LLM的字符串，并应该返回想要的
# 任何结构化数据，所有的解析逻辑都应该在这里实现
# 3.(可选)：实现get_format_instructions()方法，它应该返回一个字符串，用做给LLM的指令，告诉它
# 应该如何格式化其输出，以便parser()方法能够成功解析
# 

from langchain_core.output_parsers import BaseOutputParser
from typing import TypeVar, override


T = TypeVar("T")

class MyCustomParser(BaseOutputParser[T]):
    """
    自定义输出解析器
    """
    @override
    def parse(self, text: str) -> T:
        """
        解析LLM输出的字符串
        """
        return super().parse(text)
        

    @override
    def get_format_instructions(self) -> str:
        """
        返回给LLM的格式化指令
        """
        return super().get_format_instructions()
    
    @override
    def _type(self) -> str:
        """返回解析器的唯一类型名称"""
        return 'my_custom_parser'
        