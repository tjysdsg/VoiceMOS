a
    ņb		  �                   @   s�  d dl Z d dlmZ d dlmZ g d�Ze �d�	dd� eD ��� g d�Z
ed�	d	d� e
ddd
� D ����jZg d�Zd�	dd� eddd
� D ��Zejeed�d� e�d�Ze�d�Zdd� eD �ZW d  � n1 s�0    Y  ed� dZe �e� ed� d ZdD �]dZed7 Zeee�d � dd� eD �ZdZde d e d ZeD �]Ze�d�d Ze�d�d �d�d  Ze�d�d
 Zed k�r�ee d! e d" Z d#e d e Z!ee!ddd
� �"� ��#� d d$� ddd
� Z$d#e d e d e Z%e%�d%�d  Z&ee&ddd
� �"� ��#� d&d'� ddd
� Z'd(e$ d) e' d* Z(d+e  e d, e( Ze �e� �qT�qg d-�Ze �d�	d.d� eD ��� ed/� dS )0�    N)�md5)�ZipFile)�c   �h   �m   �o   �d   �    �a   �+   �x   r	   �.   �b   �l   �i   �z   r   r
   �r   r   � c                 C   s   g | ]}t |��qS � ��chr��.0�xr   r   �gather_test.py�
<listcomp>	   �    r   )�p   r   r   r   r   r   �s   r   c                 C   s   g | ]}t |��qS r   r   r   r   r   r   r      r   �����)�T   �A   �E   �H   �C   r    �N   �O   �D   r'   r&   r&   r    �S   r"   r    c                 C   s   g | ]}t |��qS r   r   r   r   r   r   r      r   zutf-8)�pwdz
gather.scp�   
c                 C   s   g | ]}|� d ��qS )�ascii)�decoder   r   r   r   r      r   z  ===== Gathering samples ===== 
zmkdir -p gatheredz gathering data, please wait.....)�2019�   z/1c                 C   s&   g | ]}|� d �d dt kr|�qS )�-r   �BC)�split�y)r   �kr   r   r   r   +   r   �1z#.blizzard/blizzard_wavs_and_scores_Z_release_version_�/r/   �   �_r-   z/submission_directory/z/celebrity/wav/r0   �   �.�   �   �sysz-uttz.wavzcp z
 gathered/)r   r   r   r   r   r	   r
   �-   r   r	   r   r   r   r   r   r   r
   r   r   c                 C   s   g | ]}t |��qS r   r   r   r   r   r   r   A   r   Zdone))�osZhashlibr   ZhlmZzipfiler   Zzf�cmd�system�join�a�z�o�pZsetpassword�bytes�read�fr1   �lZkl�print�cr2   �strZkp�vZbdr3   �t�gZuidZwdZsid�encodeZ	hexdigestZshZfwn0ZfwnZuhZhnr   r   r   r   �<module>   sP   
$

,



,,