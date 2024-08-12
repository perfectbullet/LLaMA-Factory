from setuptools import setup, find_packages
import deal_docx_info

requirements = [
    'python-docx',
]

setup(
    name='deal_docx_info',
    version=deal_docx_info.__version__,
    python_requires='>=3.8',
    author='zhoujing GXKJ',
    author_email='zhoujing@gx.com',
    url='https://perfectbullet.github.io/',
    description='demo of setup.py',
    license='MIT-0',
    packages=find_packages(),
    zip_safe=True,
    install_requires=requirements,
)
