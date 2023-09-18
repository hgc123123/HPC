##Partition QOS 
A QOS can be attached to a partition. This means a partition will have all the same limits as a QOS. 
This also gives the ability of a true 'floating' partition, meaning if you assign all the nodes to a 
partition and then in the Partition's QOS limit the number of GrpTRES the partition will have access 
to all the nodes, but only be able to run on the number of resources in it.

The Partition QOS will override the job's QOS. If the opposite is desired you need to have the job's 
QOS have the 'OverPartQOS' flag which will reverse the order of precedence.
