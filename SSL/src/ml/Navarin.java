package ml;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

import org.deckfour.xes.extension.std.XConceptExtension;
import org.deckfour.xes.extension.std.XLifecycleExtension;
import org.deckfour.xes.extension.std.XOrganizationalExtension;
import org.deckfour.xes.extension.std.XTimeExtension;
import org.deckfour.xes.model.XEvent;
import org.deckfour.xes.model.XLog;
import org.deckfour.xes.model.XTrace;

import parser.Parser;
import reducer.ReducedLogContainer;
import util.Util;

/**
 * This class generates all necessary input file for machine learning
 * in the python program that can be found at https://github.com/nickgentoo/DALSTM_PM.
 * 
 * 
 * 
 * 
 * @author Martin Käppel
 * @version	09.11.2020
 *
 */

public class Navarin implements Approach {
	private String targetFormat = "yyyy-MM-dd'T'HH:mm:ss.SSSXXX";
	private SimpleDateFormat sdf = new SimpleDateFormat(targetFormat);



	public void createInputFile(XLog log, List<String> eventAttributes, String path, String title, boolean lifecycle) {
		int counter = 1;
		//Create header of the csv File
		StringBuilder sb = new StringBuilder();
		sb.append("CaseID");
		sb.append(",");

		for(int i = 0; i < eventAttributes.size(); i++) {
			sb.append(eventAttributes.get(i));
			if(i != eventAttributes.size()-1) {
				sb.append(",");
			}
		}
		sb.append("\n");
		
		//Transform the traces
		Iterator<XTrace> logIterator = log.iterator();
		while(logIterator.hasNext()) {
			XTrace currentTrace = logIterator.next();
					
			Iterator<XEvent> traceIterator = currentTrace.iterator();
			while(traceIterator.hasNext()) {
				sb.append(counter);
				sb.append(",");
				XEvent currentEvent = traceIterator.next();
				//Considering activity sweparately because its intertwined with the lifecycle
				if(lifecycle == true) {
					String activity = currentEvent.getAttributes().get(eventAttributes.get(0)).toString();
					String transition = currentEvent.getAttributes().get(XLifecycleExtension.KEY_TRANSITION).toString();
					
					sb.append(activity+"-"+transition);
					sb.append(",");
				}
				else {
					sb.append(currentEvent.getAttributes().get(eventAttributes.get(0)));
				}
				//Timestamp separately to transform in the right format
				Date extractedDate = Util.tryParse(currentEvent.getAttributes().get(XTimeExtension.KEY_TIMESTAMP).toString());
				sb.append(sdf.format(extractedDate));
				sb.append(",");
				
				for(int i = 2; i < eventAttributes.size(); i++) {
					if(currentEvent.getAttributes().get(eventAttributes.get(i)) != null) {
						sb.append(currentEvent.getAttributes().get(eventAttributes.get(i)));
					}
					else {
						sb.append("");
					}
					if(i != eventAttributes.size()-1) {
						sb.append(",");
					}
				}
				sb.append("\n");
			}
			counter++;
					
		}
		
		//Serializing
		File inputFile = new File(path+"\\inp_"+title+".csv");
		if(inputFile.exists()) {
			inputFile.delete();
		}
		inputFile = new File(path+"\\inp_"+title+".csv");
				
		try {
			FileWriter fileWriter = new FileWriter(inputFile, true);
			BufferedWriter writer = new BufferedWriter(fileWriter);
			PrintWriter out = new PrintWriter(writer);
			out.print(sb.toString());
			out.close();
		}
		catch(IOException e) {
			e.printStackTrace();
		}
	}
	
	@Override
	public void createInputFiles(XLog referenceLog, Set<ReducedLogContainer> logs, String path, boolean lifecycle) {
		Parser p = new Parser();
		Set<String> eventAttributes = p.getEventAttributes(referenceLog);
		
		System.out.println(eventAttributes);
		
		//Sort the list so the attributes have the right order for the python file
		List<String> sortedEventAttributes = new ArrayList<String>();
		sortedEventAttributes.add(XConceptExtension.KEY_NAME);
		eventAttributes.remove(XConceptExtension.KEY_NAME);
		sortedEventAttributes.add(XTimeExtension.KEY_TIMESTAMP);
		eventAttributes.remove(XTimeExtension.KEY_TIMESTAMP);		
		//Remove lifecyle colum, since in case of considering lifecycle it is combined with activity otherwise it is not relevant
		eventAttributes.remove(XLifecycleExtension.KEY_TRANSITION);
		
		sortedEventAttributes.addAll(eventAttributes);

		System.out.println(sortedEventAttributes);
		
		
		createInputFile(referenceLog, sortedEventAttributes, path, "referenceLog", lifecycle);
		for(ReducedLogContainer container : logs) {
			createInputFile(container.getLog(), sortedEventAttributes, path, container.getTitle(), lifecycle);
		}
	}
}
