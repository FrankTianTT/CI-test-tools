from setuptools import setup, find_packages


def readme():
    with open('README.md') as f:
        return f.read()


setup(name='CI-test-tools',
      version='0.01',
      description='Conditional test tools implement in Python.',
      url='https://github.com/FrankTianTT/CI-test-tools',
      author='Honglong Tian',
      author_email='franktian424@gmail.com',
      license='MIT License',
      packages=find_packages())
