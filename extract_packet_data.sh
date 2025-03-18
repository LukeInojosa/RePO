#!/bin/bash

cat RePO/data/packet_based/monday.zip_parta* > RePO/data/packet_based/monday.zip
cat RePO/data/packet_based/tuesday.zip_parta* > RePO/data/packet_based/tuesday.zip
cat RePO/data/packet_based/wednesday.zip_parta* > RePO/data/packet_based/wednesday.zip

unzip RePO/data/packet_based/monday.zip -d RePO/data/packet_based
unzip RePO/data/packet_based/tuesday.zip -d RePO/data/packet_based
unzip RePO/data/packet_based/wednesday.zip -d RePO/data/packet_based
unzip RePO/data/packet_based/thursday.zip -d RePO/data/packet_based
unzip RePO/data/packet_based/friday.zip -d RePO/data/packet_based
