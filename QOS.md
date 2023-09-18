## Partition QOS 
A QOS can be attached to a partition. This means a partition will have all the same limits as a QOS. 
This also gives the ability of a true 'floating' partition, meaning if you assign all the nodes to a 
partition and then in the Partition's QOS limit the number of GrpTRES the partition will have access 
to all the nodes, but only be able to run on the number of resources in it.

The Partition QOS will override the job's QOS. If the opposite is desired you need to have the job's 
QOS have the 'OverPartQOS' flag which will reverse the order of precedence.

### how to delete a qos

```
sacctmgr delete qos zebra
```

To  clear a  previously set value use the modify command with a new value of -1 for each TRES id.

```
saccmt modify qos zebra set GrpTRES=cpu=-1
```

### How to add QOS in job_submit.lua
```
if (job_desc.partition == "CPUGPUOnlyForManager")
then
    slurm.log_info("The job will go to GPU partition")
    if (job_desc.gres == nil)
    then
	slurm.log_info("User submit jobs with the CPU requirement.")
	slurm.user_msg("You can use CPU resources")
	job_desc.qos="16coresforcpu"
	job_desc.features="gpu"
	return slurm.SUCCESS
    else
	slurm.log_info("User submit jobs with the GPU requirement.")
	slurm.user_msg("You can use GPU resources")
	job_desc.qos="small"
	job_desc.features="gpu"
	return slurm.SUCCESS
    end
end
```

## References

- [QOS](https://slurm.schedmd.com/qos.html)
