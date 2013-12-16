#!/usr/bin/R

FileName = "sts.native.sum"
Data = read.table(FileName)
# Data[,1]
PDBID = gsub("./pdb_1515_train_set/(.{6}).native.sum","\\1",Data[,1])
Data.Sum = apply(Data[,2:ncol(Data)],1,sum)
FileName = "sort.list"
Data = read.table(FileName)
Data.Length = Data[,3]

# postscript("test.ps",width=7,height=5,colormodel="cmyk")
pdf("test.pdf",width=7,height=5,colormodel="cmyk")
par(cex=1.3)
plot(Data.Length, Data.Sum, xlim=c(0,500),ylim=c(0,1500), xlab="Residue length", ylab="Number of contacts")

line.x = range(Data.Length)
line.y = 3.09 * line.x + (-76.182)
lines(line.x,line.y,col="gray",lwd=2.2)
dev.off()



# After running R script, we can get one test.pdf file. then in command line, type "pdf2ps"
# command to transform pdf to ps file. Once ps file generated, we could use "ps2eps test.ps"
# to generate test.eps file. sometimes the bouding box can not be correctly genearted, we
# could try other option "ps2eps -B -f test.ps" which can ignore the default option bbox and
# generate new bounding box. 
# resize eps file to satisfy journal rule. 1 inch = 72 points
#   epsffit -c 0 0 261 162 test.eps new.eps
# How to know the bounding box size
#   In the file.eps header include bounding box information
